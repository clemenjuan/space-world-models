# Space World Models

Minimal research repo for EventSat autonomous-operations world models, probes, and latent MPC planners.

## Repository Map

- `core/`: shared data, model, and training primitives.
- `swm_eventsat/`: EventSat schema, data adapters, models, probes, planning, configs, and experiment entry points.
- `swm_eventsat/experiments/`: runnable dataset, training, evaluation, probe, and artifact scripts.
- `swm_eventsat/config/`: Hydra configs for toy, lite, and AUTOPS EventSat runs.
- `docs/`: research tracker and method notes.
- `scripts/`: small utilities such as the local result-board server.
- `tests/`: smoke and layout regression tests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Minimal Example

Generate a tiny toy EventSat trajectory dataset:

```bash
python -m swm_eventsat.experiments.generate_dataset \
  --n-episodes 4 \
  --episode-len 32 \
  --out data/cache/eventsat_toy_smoke.npz
```

For the current AUTOPS/EventSat research pipeline, see `docs/research_tracker.md`.

## Result Board

```bash
python scripts/serve_board.py
```

The board serves `data/figures/index.html` at `http://127.0.0.1:8801/`.
