# Step 3 — Single-Satellite EO Scheduling Environment wired to LeWM

**Date:** 2026-06-17
**Status:** Approved design (pending spec review)
**Scope:** One Gymnasium EO-scheduling environment + a random-policy baseline + a LeWM
forward-pass wiring, reusing the Step 1/2 model code **with no model-code changes**.
Environment and forward pass only — no trained/learned policy, no training loop.

---

## 1. Goal

Add a single-satellite Earth-observation (EO) scheduling environment. The satellite has
imaging opportunities (access windows) precomputed from a propagated Keplerian/SSO orbit,
a queue of imaging requests arriving stochastically, and on-board storage and power that
evolve under a simple linear model. A random policy decides whether to image at each step.
The LeWM encoder/predictor is wired for a forward pass over the 6-dim observation, but is
**not trained** in this step.

## 2. Reuse contract (no model changes)

- `models.od_encoder.OdEncoder(in_dim=6, hidden_dim=256, out_dim=192)` — parameterized by
  `in_dim`, no code change.
- `models.od_jepa.ODJEPA` — `encode()` already reads `info["obs"]` dim-agnostically.
- `module.ARPredictor`, `module.Embedder(input_dim=2)` (one-hot of binary action),
  `module.SIGReg`, `module.MLP` — unchanged.
- `embed_dim = 192`, `history_size = 3` — same as Step 1/2.
- `envs.orekit_setup.ensure_orekit` and the EcksteinHechler propagator + `TopocentricFrame`
  pattern from `envs/od_env.py` — reused (not modified).

A **new additive** `models/factory.py` generalizes the existing `_make_odjepa` recipe into
`build_jepa(obs_dim, action_dim, ...)`. It does not modify any existing model file. Step 1/2
keep their inline model construction.

## 3. Repo layout (new files)

```
envs/access_windows.py       # Orekit window precompute (propagate SSO + target access -> window timeline)
envs/scheduling_env.py       # Gymnasium env: windows + request queue + linear storage/power
models/factory.py            # build_jepa(obs_dim, action_dim, embed_dim=192, history=3) — additive, DRY
agents/__init__.py
agents/random_policy.py      # run_random_policy(env, n_episodes, seed) -> per-episode rewards, logs reward
tests/test_scheduling.py     # env contract + bounded storage/power + model forward + baseline reward
```

No changes to: `module.py`, `jepa.py`, `models/od_encoder.py`, `models/od_jepa.py`,
`models/od_forward.py`, `envs/od_env.py`, `envs/orekit_setup.py`, `envs/fdir_env.py`.

## 4. `envs/access_windows.py`

Precompute the access-window timeline once (called from `reset`):
- Propagate the SSO orbit (same params as Step 1: `a=6778137 m`, `e=1e-3`, `i=97.0 deg`,
  EcksteinHechler J2-J6) over the episode horizon at step resolution `dt = 30 s`.
- For a small fixed set of ground targets (configurable `GeodeticPoint`s, default a handful
  of mid-latitude sites), compute satellite elevation per step via `TopocentricFrame`.
- Mark step `k` as **in-window** if any target elevation > a mask angle (default 5 deg).
- Grid-sampled (robust; avoids fragile Orekit event detection).
- Provide: `windows: np.ndarray[bool]` of length `horizon`, and a helper
  `time_to_next_window(k) -> seconds` (0 while in a window; large/horizon-capped if none
  ahead).

`dt = 30 s`, horizon >= 500 steps (≈ 4.2 h ≈ 2.7 orbits) so several windows occur.

## 5. `envs/scheduling_env.py`

Gymnasium env mirroring the `OdEnv` interface.

### 5.1 Hidden state
- `storage_fill in [0, 1]`, `power_level in [0, 1]` (start near a configurable level).
- A pending-request queue: list of priorities (float/int).
- `step` counter; precomputed `windows` from `access_windows`.

### 5.2 Requests (seeded stochastic)
- Each step, `n ~ Poisson(lambda)` new requests arrive (default lambda small, e.g. 0.2),
  each with a random priority (default integer 1-5, uniform).
- Seeded via Gymnasium `reset(seed=...)` RNG -> deterministic per seed, varied across seeds.
- `priority_of_next_request` = max priority in queue (0 if empty).

### 5.3 Observation (6-dim, raw units)
`[orbit_phase (0-1), time_to_next_window (s), storage_fill (0-1), power_level (0-1),
n_pending_requests, priority_of_next_request]`.
- `orbit_phase = (t mod T_orbit) / T_orbit`.
- `observation_space = Box(shape=(6,))` with generous bounds.

### 5.4 Action
- `action_space = gym.spaces.Discrete(2)` {0=skip, 1=image}.
- One-hot encoding to a 2-vector happens at model-feed time (factory/baseline/test), not in
  the env. (Noted: extends to a 3-dim action for multi-satellite in a later step.)
- Action does not change orbit/windows; it gates imaging only.

### 5.5 Linear dynamics & reward
Per `step(action)`:
1. Arrivals: add Poisson requests to the queue.
2. Passive update: `power_level += charge_rate - base_drain`; `storage_fill -= downlink_rate`;
   clamp both to `[0, 1]`.
3. If `action == 1`: an image is **successful** iff in-window AND queue non-empty AND
   `storage_fill + image_size <= 1` AND `power_level - image_drain >= 0`:
   - success -> dequeue highest-priority request; `reward += priority`;
     `storage_fill += image_size`; `power_level -= image_drain` (then clamp).
   - if in-window AND queue non-empty but storage would overflow OR power would be violated
     -> `reward = -1` (image not taken; state stays clamped).
   - otherwise (no window / empty queue) -> no-op, `reward = 0`.
4. `action == 0` -> `reward = 0`.
Storage and power are **hard-clamped to [0, 1]**, so they remain bounded by construction;
the -1 penalty is what flags an attempted overflow/power violation.

### 5.6 Termination & info
- `truncated = step >= max_steps` (default 500); `terminated = False`.
- `info` includes the true `state` (storage, power), `in_window` (bool), `queue_len`.

## 6. `models/factory.py` — forward-pass wiring

```
build_jepa(obs_dim, action_dim, embed_dim=192, history=3, hidden_dim=256) -> ODJEPA
```
Assembles `OdEncoder(in_dim=obs_dim, out_dim=embed_dim)`, `ARPredictor(num_frames=history,
input_dim=embed_dim, ...)`, `Embedder(input_dim=action_dim, emb_dim=embed_dim)`,
`MLP` projector + pred_proj — the existing `_make_odjepa` recipe, generalized. Step 3 uses
`build_jepa(6, 2)`. **Forward pass only**: `encode` + `predict`; no training, no loss step.

## 7. `agents/random_policy.py`

`run_random_policy(env, n_episodes=1, seed=0) -> list[float]`:
- Runs `n_episodes`, sampling `env.action_space.sample()` each step until truncated.
- Accumulates per-episode reward, prints each episode's total reward to stdout, returns the
  list. (W&B optional and off by default; no training.)

## 8. Smoke test — `tests/test_scheduling.py`

1. **Bounded storage/power:** 500-step rollout under the random policy; assert every step
   `0 <= storage_fill <= 1` and `0 <= power_level <= 1`; obs shape `(6,)`, all finite.
2. **Window sanity:** the precomputed timeline contains >= 1 access window over the horizon
   (so imaging is sometimes possible).
3. **Model forward:** `build_jepa(6, 2)` -> encode `(1, T, 6)` obs + `(1, T, 2)` one-hot
   action -> `emb (1, T, 192)`; `predict` -> `(1, T, 192)`; finite.
4. **Baseline reward:** `run_random_policy` returns finite per-episode rewards.

## 9. Out of scope

- Any trained / learned scheduling policy, RL training, or world-model training in this step.
- Reward shaping beyond the spec (+priority / -1).
- Multi-satellite (3-dim action) and inter-satellite coordination.
- Downlink-window modeling (constant `downlink_rate` drain only).
- Cloud cover, agility/slew constraints, data-latency objectives.
