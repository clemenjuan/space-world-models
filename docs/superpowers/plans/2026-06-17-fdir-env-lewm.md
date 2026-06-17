# Step 2 — FDIR Environment wired to LeWM — Implementation Plan

**Date:** 2026-06-17
**Spec:** `docs/superpowers/specs/2026-06-17-fdir-env-design.md` (approved)
**Workflow:** subagent-driven (implementer → review → commit), trunk-based (commit each task to `main`).

## Goal recap
Add a fault-detection (FDIR) environment. Telemetry evolves under a stable linear model;
faults injected at the **state level** propagate through coupled channels. The Step-1 LeWM
model (encoder + predictor) is trained self-supervised on **nominal-only** telemetry, then
anomalies are flagged by high prediction error:
`surprise_score_t = || z_hat_t - enc(o_t) ||^2`. Detection only — no recovery policy.

## Reuse contract (NO model-code changes — verified)
- `models.od_encoder.OdEncoder` — already takes `in_dim`; use `in_dim=8`. (`forward` is dim-agnostic.)
- `models.od_jepa.ODJEPA.encode` — reads `info["obs"]`, dim-agnostic. Unchanged.
- `module.ARPredictor` (verbatim, `**kwargs` as in `config/train/model/od.yaml`),
  `module.Embedder(input_dim=4, smoothed_dim=4, emb_dim=192)`, `module.SIGReg(knots, num_proj)`,
  `module.MLP`. Unchanged.
- `models.od_forward.od_lejepa_forward` — unchanged (`pred MSE + 0.09*SIGReg`).
- `od_datasets.od_dataset.OdWindowDataset` + `fit_normalizers` — already dim-agnostic
  (`.shape[-1]`); reused unchanged by storing the action as a one-hot `(N,L,4)` array in the
  `.npz` (nominal action 0 → one-hot `[1,0,0,0]`; constant columns → std guarded to 1).
- `spt_compat` shims (incl. `stub_lance_if_no_avx`) + harness pattern from `train_od.py`.
- `embed_dim=192`, `history_size=3`, `num_preds=1`.

**No edits** to: `module.py`, `jepa.py`, `models/od_encoder.py`, `models/od_jepa.py`,
`models/od_forward.py`, `od_datasets/od_dataset.py`, `envs/od_env.py`, `spt_compat.py`.

## New files
```
envs/fdir_env.py             # 8-dim linear telemetry env + state-level fault injection
models/surprise.py           # surprise_score(model, obs_seq, action_seq) -> per-step ||z_hat-enc(o)||^2
data/generate_fdir.py        # roll out N nominal episodes (no fault, action 0) -> .npz (obs, action one-hot, state)
train_fdir.py                # analogue of train_od.py (in_dim=8, action input_dim=4)
config/train/fdir.yaml       # top-level (mirrors od.yaml; project default WANDB_PROJECT)
config/train/model/fdir.yaml # mirrors model/od.yaml with encoder.in_dim=8, action_encoder.input_dim=4/smoothed_dim=4
config/train/data/fdir.yaml  # path data/cache/fdir_trajectories.npz, window, batch
tests/test_fdir.py           # env contract + surprise-spike detection (accreted across tasks)
```

---

## Env design constants (module-level, configurable)
8-dim state `[solar_array_voltage, battery_soc, panel_temp, obc_temp, rw_speed_x,
rw_speed_y, rw_speed_z, bus_current]`. Operating point `x*` at realistic setpoints, e.g.
`[28.0 V, 0.80, 40.0 C, 25.0 C, 1000, -500, 800 rpm, 5.0 A]`.

Nominal dynamics on deviation `d = x - x*`: `d_{t+1} = A d + w`, `w ~ N(0, Q)`.
- `A`: near-diagonal, diagonal entries in ~[0.80, 0.95] (mean-reverting), diagonally dominant
  so spectral radius < 1 (assert in a test). Documented off-diagonal couplings (~0.05–0.10):
  `solar_array_voltage→battery_soc`, `solar_array_voltage→panel_temp`,
  `rw_speed_{x,y,z}→bus_current`.
- `Q`: small diagonal process noise per channel (units-appropriate).
- Sensor noise `sigma`: per-channel, small relative to nominal deviation std.

Fault injection (state-level, active when `t >= fault_step`), `fault_mode ∈
{None,"stuck_at","drift","spike"}`, `fault_channel`, mode params:
- `stuck_at`: override that channel's update to hold its onset value; couplings still feed neighbours.
- `drift`: add constant per-step increment to the channel (ramp).
- `spike`: add transient impulse to the channel for `spike_duration` steps.

Obs `o_t = x_t + sensor_noise`. `observation_space = Box(shape=(8,))` raw units.
`action_space = Discrete(4)` {0 nominal,1 isolate_power,2 safe_mode,3 reset_obc}; action does
**not** affect dynamics; one-hot at model-feed time. `info` carries true `state` (8,) and
`fault_active` bool. Gym API mirrors `OdEnv` (`reset(seed=...)`, `step(action)`).

---

## Task 1 — FDIR environment + env-contract test
**Files:** `envs/fdir_env.py` (new), `tests/test_fdir.py` (new, env-contract test).
**Build:** `FdirEnv(gym.Env)` per the design constants above. Seeded via Gymnasium
`reset(seed=...)` (`self.np_random`). Faults modify state evolution, not just the emission.
**Test (`test_fdir_env_contract`):** rollout ≥150 steps;
- obs shape `(8,)` each step, all finite; `observation_space.shape == (8,)`.
- `action_space == Discrete(4)`.
- spectral radius of `A` < 1.
- a `stuck_at` fault (channel e.g. `solar_array_voltage`, `fault_step=100`) yields a state
  trajectory that **diverges** from the nominal (no-fault, same-seed) trajectory after step 100
  (e.g. max abs deviation on the faulted/coupled channels grows materially), and matches it
  before 100 (same seed → identical pre-fault states).
**Verify:** `python -m pytest tests/test_fdir.py -v`.

## Task 2 — Surprise metric + unit test
**Files:** `models/surprise.py` (new), `tests/test_fdir.py` (append `test_surprise_shapes`).
**Build:** `surprise_score(model, obs_seq, action_seq) -> Tensor[T - history_size]`.
- `obs_seq (1,T,8)`, `action_seq (1,T,4)` one-hot. `emb = model.encode({"obs":...,"action":...})["emb"]`.
- For each `t in [history_size, T)`: predict `z_hat_t` from the embedding window
  `emb[:, t-history_size:t]` and matching action window, using `model.predict` **exactly as
  `od_lejepa_forward` calls it** (take the last predicted step), then `||z_hat_t - emb[:,t]||^2`.
- Returns per-step scores aligned to absolute steps `history_size .. T-1`. Pure inference
  (`torch.no_grad()`); no grads, no new loss terms.
**Test:** build a tiny `ODJEPA` (in_dim=8, action input_dim=4) via a local helper mirroring
`tests/test_model._make_odjepa`; assert returned shape `== (T - history_size,)` and finite.
**Verify:** `python -m pytest tests/test_fdir.py -v`.

## Task 3 — Nominal dataset generation (reuse OdWindowDataset)
**Files:** `data/generate_fdir.py` (new), `tests/test_fdir.py` (append `test_fdir_dataset`).
**Build:** `generate(n_episodes, episode_len, out_path, seed)` rolls out N nominal episodes
(`fault_mode=None`, action 0). Saves `obs (N,L,8)`, `action (N,L,4)` **one-hot of action 0 =
[1,0,0,0]**, `state (N,L,8)` to `data/cache/fdir_trajectories.npz`. CLI mirrors
`data/generate_dataset.py` (`--n-episodes --episode-len --out --seed`).
**Test:** generate tiny set; assert npz shapes `(n,L,8)`/`(n,L,4)`; then
`OdWindowDataset(path, window=4, normalizers=fit_normalizers(path))[0]` yields
`obs (4,8)` + `action (4,4)` (existing dataset reused unchanged, proving dim-agnosticism).
**Verify:** `python -m pytest tests/test_fdir.py -v`.

## Task 4 — Training harness + configs + detection smoke test
**Files:** `train_fdir.py` (new), `config/train/fdir.yaml`, `config/train/model/fdir.yaml`,
`config/train/data/fdir.yaml` (new), `tests/test_fdir.py` (append `test_fdir_detection`).
**Build:** `train_fdir.py` mirrors `train_od.py` exactly (same spt_compat shims incl.
`stub_lance_if_no_avx`, `spt.Module`/`DataModule`/`Manager`, AdamW + LinearWarmupCosineAnnealingLR,
CPU-safe trainer, W&B gated by `cfg.wandb.enabled`). Configs mirror `od*.yaml` changing only:
`model.encoder.in_dim=8`, `model.action_encoder.input_dim=4` & `smoothed_dim=4`,
`data.path=data/cache/fdir_trajectories.npz`, `config_name="fdir"`.
**Test (`test_fdir_detection`, deterministic via fixed seeds):** generate a small nominal
dataset → train briefly (a few epochs, tiny model, CPU, `wandb.enabled` off / `logger=False`,
seconds) → one ≥150-step rollout with `stuck_at` fault at step 100 → build one-hot action seq
→ `surprise_score` → assert post-fault window mean **significantly** exceeds pre-fault:
`mean(post) > mean(pre) + k*std(pre)` (k≈3). Relative threshold (post vs pre), not absolute.
**Verify:** `python -m pytest tests/test_fdir.py -v` and full `python -m pytest -v`
(Step-1 11 tests + FDIR tests all pass). Optionally a real short `python train_fdir.py` smoke
(W&B disabled) before any real run.

---

## Out of scope (per spec)
Recovery/action policy, supervised/labeled-fault training, any world-model code change,
multi-fault scenarios, coupling the discrete action to dynamics.

## Verification gates per task
1. New/updated tests pass (`pytest tests/test_fdir.py -v`).
2. No edits to the frozen files listed above (`git diff --stat` review).
3. Full suite green (`python -m pytest -v`).
4. One commit per task to `main` with a descriptive message.
