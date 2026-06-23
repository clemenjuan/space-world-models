# Space World Models

Minimal research repo for EventSat scheduling with latent world models and MPC.

Active code:
- `core/`: shared vector world-model primitives.
- `swm_eventsat/`: EventSat data adapters, probes, planners, AUTOPS bridges, artifacts, configs, and experiments.

Legacy code:
- `legacy/od/` and `legacy/fdir/` contain the previous OD and FDIR tracks.
- `legacy/compat/` contains old flat import wrappers and retired entrypoints.
- `legacy/artifacts/` contains old OD/FDIR generated outputs.

Board:
```bash
python scripts/serve_board.py
```

The active results landing page is served on port `8801` by default:
`http://127.0.0.1:8801/`.

The direct EventSat board is still available at
`http://127.0.0.1:8801/eventsat_results_board.html`, and archived OD/FDIR result boards are linked from the landing page.
