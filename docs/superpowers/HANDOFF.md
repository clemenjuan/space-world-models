# HANDOFF — Step 1: OD environment wired to LeWM

**Purpose:** Resume context for any agent (Claude, Codex, etc.) continuing this work if the
current session hits its limit. Keep this file updated as tasks complete.

## What we're building
Adapt LeWorldModel (JEPA world model) to satellite orbit determination. Step 1 = one
Gymnasium OD env (Orekit J2 dynamics) + LeWM encoder/predictor/SIGReg + full Lightning
training loop + smoke tests. No scheduling (Step 2), no FDIR (Step 3).

## Authoritative documents (read these first)
- **Spec:** `docs/superpowers/specs/2026-06-17-od-env-lewm-design.md`
- **Plan (task-by-task, with full code):** `docs/superpowers/plans/2026-06-17-od-env-lewm.md`
- **Baseline being adapted:** https://github.com/lucas-maes/le-wm
  (`module.py` = SIGReg/ARPredictor/Embedder/MLP; `jepa.py` = JEPA; `train.py` = loss + harness)

## Workflow in use
Subagent-driven development (superpowers skill): per task, dispatch implementer →
spec-compliance review → code-quality review → mark complete. Branch: `step1-od-env`
(off `master`). Each task ends with its own commit (see plan for messages).

## Environment facts (verified live)
- Windows 11, Python 3.12.6 at `C:\Python312\python.exe`. PowerShell + Git Bash available.
- Java 16 present (`JAVA_HOME` set) → `orekit_jpype` works.
- `orekit_jpype`, `gymnasium`, `numpy` ALREADY installed. `orekit-data.zip` ALREADY
  downloaded at repo root (gitignored, ~70 MB).
- NOT yet installed: torch, einops, stable_pretraining, stable_worldmodel, lightning,
  hydra-core, wandb, pytest (Task 0 installs these via `requirements.txt`).
- W&B: entity `sps-tum`, project from `$WANDB_PROJECT` (default `space-world-models`).
  User must `wandb login`; tests run with wandb disabled.

## Key gotchas (cost real time to rediscover)
- Orekit: call `orekit_jpype.initVM()` BEFORE importing any `org.orekit.*`/`java.*`. Do it
  once per process — see `envs/orekit_setup.py` (`ensure_orekit()`, idempotent).
- Constants: `Constants.EIGEN5C_EARTH_MU = 3.986004415e14`, `..._EQUATORIAL_RADIUS = 6378136.46`,
  `J2 = -Constants.EIGEN5C_EARTH_C20 = 1.0826e-3`.
- Propagator: `EcksteinHechlerPropagator(orbit, Re, mu, C20, C30, C40, C50, C60)`.
- Range-rate: relative PV; station PV via
  `topo.getTransformTo(eci, date).transformPVCoordinates(PVCoordinates.ZERO)`.
- Measured invariant drift over 200×30 s SSO@400 km rollout: rel ΔE 4.5e-5, rel Δh_z 2.2e-5,
  RAAN rate 0.981°/day. Test tolerances: ΔE/Δh_z `<1e-3`, RAAN in `[0.95,1.02]`.
- Only `JEPA.encode()` is overridden (pixels→obs + MLP encoder). Predictor + SIGReg are
  used VERBATIM from vendored `module.py`/`jepa.py`.
- ONLY unverified integration surface: `stable_pretraining.Module`/`Manager` API (0.1.7).
  `train_od.py` mirrors `le-wm/train.py`; adapt call sites if signatures differ.

## Task status
- [x] Task 0 — project setup + vendored baseline (requirements, module.py, jepa.py, __init__)
- [x] Task 1 — Orekit bootstrap (`envs/orekit_setup.py`)
- [x] Task 2 — OdEnv dynamics core
- [x] Task 3 — OdEnv measurement model
- [x] Task 4 — 200-step rollout shape contract
- [x] Task 5 — invariants (J2 energy, h_z, SSO RAAN precession)  [tests/test_env.py: 6 passed]
- [ ] Task 6 — OdEncoder MLP
- [ ] Task 7 — ODJEPA encode() override
- [ ] Task 8 — od_lejepa_forward loss
- [ ] Task 9 — dataset generation + windowed Dataset
- [ ] Task 10 — configs + train_od.py + W&B + train smoke test

## Notes for resumers
- Env tests need the JVM up before importing `org.orekit.*`. If a test imports Constants
  directly, call `ensure_orekit()` first (see `test_invariants_bounded`).
- Harmless JVM-teardown traceback prints at pytest process exit; `6 passed` still reported.

## To resume
1. `git checkout step1-od-env`; check `git log --oneline` to see which task commits exist.
2. Open the plan; find the first unchecked task; execute its steps (TDD).
3. `pytest -v` should pass for all completed tasks.
4. Update the checkboxes above after each task.
