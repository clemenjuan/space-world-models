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
spec-compliance review -> code-quality review -> mark complete.
**TRUNK-BASED ONLY: commit directly to `main` and push; do NOT create feature branches.**
(User directive, 2026-06-17.) One commit per logical task.

## Project status across steps (2026-06-17)
- **Step 1 (OD env) — BUILT & trained-smoke-verified.** Code on `main`, 11 tests pass.
- **Step 2 (FDIR env) — SPEC ONLY.** `docs/superpowers/specs/2026-06-17-fdir-env-design.md`.
  Not implemented. Train-on-nominal then surprise = `||z_hat - enc(o)||^2`; state-level faults.
- **Step 3 (EO scheduling env) — SPEC ONLY.**
  `docs/superpowers/specs/2026-06-17-scheduling-env-design.md`. Not implemented. Orekit
  access windows + Poisson requests + linear storage/power; env + forward pass only.
- **Next action:** moving compute to the TUM VM (see "Running on the VM" below). First goal:
  reproduce a real OD training run on the VM, THEN implement Step 2, then Step 3.

## Running on the VM (Linux bootstrap) — autops-demo-clemente
24 vCPU / 48 GB Ubuntu 24.04, no GPU (fine — models are ~8M params, CPU-trainable; the real
cost is CPU-bound Orekit data gen, which parallelizes across the 24 cores). To set up:
1. `sudo apt update && sudo apt install -y python3.12-venv default-jre git` (Java is required
   by `orekit_jpype`; Ubuntu 24.04 default-jre = Java 21, works with orekit_jpype 13.x).
2. `git clone https://github.com/clemenjuan/space-world-models.git && cd space-world-models`
3. `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
4. First run auto-downloads `orekit-data.zip` via `ensure_orekit()` (needs internet; the TUM
   network has it).
5. `wandb login` (entity `sps-tum`, project `space-world-models`).
6. Verify: `python -m pytest -v` should report all Step 1 tests passing.
7. On Linux the `spt_compat` Windows signal/stdio shims are no-ops; the pyarrow patch may
   still apply depending on installed `datasets`/`pyarrow` versions — keep calling it.
8. **No-AVX CPU (TUM VM is `qemu64`, flags = `sse4_1 sse4_2` only):** `pylance`'s native
   ext raises an *uncatchable* SIGILL ("Illegal instruction") on import. `lance` is imported
   only by `stable_pretraining.data.video` (unused by our vector-obs tasks), so
   `spt_compat.stub_lance_if_no_avx()` registers a stub `lance` module before importing
   `stable_pretraining` (gated on AVX absent from `/proc/cpuinfo`; no-op on real CPUs).
   Called by `train_od.py` and `tests/test_train_smoke.py`. The real fix is host CPU
   pass-through (`-cpu host`) at the hypervisor; torch/numpy/pyarrow already work on SSE4.

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
1. On `main` (trunk-based; no branches). `git pull` then `git log --oneline` for current state.
2. Bootstrap the VM per "Running on the VM" above; confirm `python -m pytest -v` passes.
3. First milestone: a real OD training run on the VM (generate a real dataset, then
   `python train_od.py` with W&B enabled) and confirm the loss converges.
4. Then implement Step 2 (FDIR) and Step 3 (scheduling) from their approved specs via the
   brainstorm-done -> writing-plans -> subagent-driven flow. Commit each task to `main`.
