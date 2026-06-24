# Start Here

## What This Repo Owns

- EventSat latent world-model training.
- Probe models from latent state to mission attributes.
- Latent utility and CEM/MPC planning code.
- Compact planner artifacts for AUTOPS evaluation.

## Quick Setup

- Create and activate a virtual environment.
- Install `requirements.txt`.
- Run `pytest` before changing core behavior.

## Ten-Minute Toy Run

- Generate toy EventSat trajectories with `swm_eventsat.experiments.generate_dataset`.
- Keep generated datasets under `data/cache/`.
- Keep run outputs, checkpoints, and W&B files out of git.

## Core Code Map

- `core/data/`: windowed trajectory datasets and normalizers.
- `core/models/`: VectorJEPA/LeWM model components and loss.
- `swm_eventsat/data/`: EventSat toy environments and AUTOPS adapters.
- `swm_eventsat/models/`: probes, artifact schemas, checkpoint helpers, and model specs.
- `swm_eventsat/planning/`: CEM planner, action masks, and planner scoring.

## Experiment Scripts

- `swm_eventsat/experiments/generate_dataset.py`: toy EventSat trajectory generation.
- `swm_eventsat/experiments/train_world_model.py`: LeWM training entry point.
- `swm_eventsat/experiments/train_autops_probes.py`: AUTOPS probe fitting.
- `swm_eventsat/experiments/write_planner_artifact.py`: package a planner artifact for AUTOPS.

## What Not To Commit

- Virtual environments.
- `outputs/`, `wandb/`, logs, checkpoints, generated boards, and cached datasets.
- One-off notebooks, scratch scripts, and local experiment dumps.
