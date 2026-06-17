# HANDOFF - Step 1: OD environment wired to LeWM

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
Subagent-driven development (superpowers skill): per task, dispatch implementer ->
spec-compliance review -> code-quality review -> mark complete. Branch: `step1-od-env`
(off `master`). Each task ends with its own commit (see plan for messages).

## Environment facts (verified live)
- Windows 11, Python 3.12.6 at `C:\Python312\python.exe`. PowerShell + Git Bash available.
- Java 16 present (`JAVA_HOME` set) -> `orekit_jpype` works.
- `orekit_jpype`, `gymnasium`, `numpy` installed; `orekit-data.zip` downloaded at repo root
  (gitignored, about 70 MB).
- Installed during Task 0: torch, einops, stable_pretraining, stable_worldmodel, lightning,
  hydra-core, wandb, pytest.
- W&B: entity `sps-tum`, project from `$WANDB_PROJECT` (default `space-world-models`;
  project exists at https://wandb.ai/sps-tum/space-world-models). User should run
  `wandb login` or set `WANDB_API_KEY`; tests run with W&B disabled.

## Key gotchas (cost real time to rediscover)
- Orekit: call `orekit_jpype.initVM()` BEFORE importing any `org.orekit.*`/`java.*`. Do it
  once per process - see `envs/orekit_setup.py` (`ensure_orekit()`, idempotent).
- Constants: `Constants.EIGEN5C_EARTH_MU = 3.986004415e14`, `..._EQUATORIAL_RADIUS = 6378136.46`,
  `J2 = -Constants.EIGEN5C_EARTH_C20 = 1.0826e-3`.
- Propagator: `EcksteinHechlerPropagator(orbit, Re, mu, C20, C30, C40, C50, C60)`.
- Range-rate: relative PV; station PV via
  `topo.getTransformTo(eci, date).transformPVCoordinates(PVCoordinates.ZERO)`.
- Measured invariant drift over 200 x 30 s SSO@400 km rollout: rel dE 4.5e-5,
  rel dh_z 2.2e-5, RAAN rate 0.981 deg/day. Test tolerances: dE/dh_z `<1e-3`,
  RAAN in `[0.95,1.02]`.
- Only `JEPA.encode()` is overridden (pixels -> obs + MLP encoder). Predictor + SIGReg are
  used verbatim from vendored `module.py`/`jepa.py`.
- `stable_pretraining.Module`/`Manager` API (0.1.7) is verified by
  `tests/test_train_smoke.py` and a real `train_od.py` two-batch CPU run.
- Do not use a local top-level package named `datasets`: it shadows HuggingFace
  `datasets`, which `stable_pretraining` imports. OD dataset code lives in `od_datasets/`.
- Installed HuggingFace `datasets==2.14.4` with `pyarrow==24.0.0` needs
  `spt_compat.patch_pyarrow_for_legacy_datasets()` before importing `stable_pretraining`.
  Windows also needs the signal-log and UTF-8 stdio shims in `spt_compat.py`.

## Task status
- [x] Task 0 - project setup + vendored baseline (requirements, module.py, jepa.py, __init__)
- [x] Task 1 - Orekit bootstrap (`envs/orekit_setup.py`)
- [x] Task 2 - OdEnv dynamics core
- [x] Task 3 - OdEnv measurement model
- [x] Task 4 - 200-step rollout shape contract
- [x] Task 5 - invariants (J2 energy, h_z, SSO RAAN precession)
- [x] Task 6 - OdEncoder MLP
- [x] Task 7 - ODJEPA encode() override
- [x] Task 8 - od_lejepa_forward loss
- [x] Task 9 - dataset generation + windowed Dataset
- [x] Task 10 - configs + train_od.py + W&B + train smoke test

## Notes for resumers
- Env tests need the JVM up before importing `org.orekit.*`. If a test imports Constants
  directly, call `ensure_orekit()` first (see `test_invariants_bounded`).
- Harmless JVM-teardown traceback prints at pytest process exit; `11 passed` still reported.
- Verified Task 10 commands:
  - `python -m pytest tests/test_train_smoke.py::test_train_smoke -v`
  - `python data/generate_dataset.py --n-episodes 4 --episode-len 16 --out data/cache/od_trajectories.npz --seed 0`
  - `python train_od.py data.batch_size=8 trainer.max_epochs=1 trainer.accelerator=cpu trainer.devices=1 ++trainer.limit_train_batches=2 ++trainer.limit_val_batches=1 wandb.enabled=false`
  - `python -m pytest -v` -> `11 passed`

## To resume
1. `git checkout step1-od-env`; check `git log --oneline` to see which task commits exist.
2. Open the plan/spec and choose the next Step 2 work item.
3. `pytest -v` should pass for all completed Step 1 tasks.
