# Research Tracker: EventSat LeWM-MPC
*Last updated: June 23, 2026*

## Rule

No trash files. Keep one concise tracker, necessary source modules, and scripts that are actually run.

## Paper Claim

**Draft title:** Latent World Model-Based Mission Planning for Autonomous Satellite Operations

The method is an **autonomous onboard (AO)** controller for EventSat: at every timestep it observes the current spacecraft state, rolls out candidate mode sequences in a latent world model, applies the first action, and replans.

Training data may come from AO, AG, AH, symbolic, LLM, or scheduled traces. For transition learning, the action source does not matter as long as each tuple is clean:

```text
obs_t, action_t -> obs_{t+1}, state_{t+1}, reward_t
```

The evaluation claim stays AO closed-loop.

## Repo Boundary

`space-world-models` owns the research method:

- LeWM/VectorJEPA training on EventSat trajectories;
- AUTOPS dataset loading and validation;
- linear probes from latent state to mission attributes;
- latent utility and CEM-MPC planner;
- compact artifacts for AUTOPS to consume.

`autops-agentic-framework` owns the truth environment and benchmark surface:

- EventSat simulator;
- baseline planners and operations paradigms;
- world-model trace export;
- results board and metrics.

## AUTOPS Dataset Contract

Canonical dataset: `eventsat_world_model_v1`.

Required arrays:

```text
obs           float32  (episode, time, 25)
action        float32  (episode, time, 7)
state         float32  (episode, time, 25)
reward        float32  (episode, time)
mode          int64    (episode, time)
resolved_mode int64    (episode, time)
forced_mode   float32  (episode, time)
episode_seed  int64    (episode,)
episode_id    int64    (episode,) optional but preferred
```

Action is 7D, one-hot over the operational EventSat modes:

```text
0 charging
1 communication
2 payload_observe
3 payload_compress
4 payload_detect
5 payload_send
6 safe
```

AUTOPS v1 state is simulator-native. Thermal and pointing are **not** probe targets unless AUTOPS is extended to simulate them.

Useful state/probe attributes now:

- battery margin;
- storage margin;
- downlink progress;
- science observation progress;
- detection progress;
- communication opportunity;
- forced-mode risk;
- health/anomaly-safe flag.

## Current Export

The intended full export is produced in AUTOPS:

```bash
cd ~/autops-agentic-framework
uv run python scripts/export_eventsat_world_model_traces.py \
  configs/experiments/eventsat_sas_ao_symb.yaml \
  configs/experiments/eventsat_sas_ag_symb.yaml \
  configs/experiments/eventsat_sas_ah_symb_symb.yaml \
  --episodes 5 \
  --steps 10080 \
  --seed 42 \
  --out data/world_model/eventsat_autops_v1
```

Expected output:

```text
~/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz
~/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.metadata.json
```

Observed full-export shape when available:

```text
obs    (15, 10080, 25)
action (15, 10080, 7)
state  (15, 10080, 25)
```

## Method Pipeline

1. Export AUTOPS EventSat traces.
2. Validate with `swm_eventsat.schema.load_world_model_dataset`.
3. Train LeWM with 25D observations and 7D mode actions.
4. Train linear probes from frozen LeWM latents to AUTOPS mission attributes.
5. Build a planner artifact: checkpoint, normalizers, probe weights, utility weights, CEM settings.
6. Run LeWM-CEM as an AUTOPS AO representation.
7. Refresh the AUTOPS board and compare against AO symbolic/LLM/HLLM baselines.

W&B training entry point:

```bash
cd ~/space-world-models
WANDB_MODE=online \
WANDB_PROJECT=space-world-models \
WANDB_RUN_NAME=eventsat-autops-action7-lewm-full \
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  --config-name train_autops
```

Equivalent explicit smoke entry point:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  data.path=/home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  model.action_encoder.input_dim=7 \
  model.action_encoder.smoothed_dim=7 \
  trainer.max_epochs=1 \
  wandb.enabled=false
```

Probe smoke entry point, before real frozen LeWM latents are available:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --out outputs/eventsat_autops_probe_smoke.npz
```

Latent probe and planner artifact handoff after LeWM training:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.export_autops_latents \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --checkpoint /path/to/lewm.ckpt \
  --out outputs/eventsat_autops_latents.npz

.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --latents outputs/eventsat_autops_latents.npz \
  --out outputs/eventsat_autops_probe_latent.npz

.venv/bin/python -m swm_eventsat.experiments.write_planner_artifact \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --checkpoint /path/to/lewm.ckpt \
  --probe outputs/eventsat_autops_probe_latent.npz \
  --out outputs/eventsat_autops_lewm/planner_artifact.json
```

AUTOPS learned rollout requires Torch in the AUTOPS runtime. Use `uv run --extra rl autops run ...`; `strict_artifact: true` prevents silent surrogate fallback.

## Baselines

Primary comparisons should be AO because LeWM-CEM replans onboard at every timestep.

Use AG/AH traces and planners as:

- training data for transition coverage;
- cross-paradigm baseline context;
- ablations for whether ground-style schedules improve world-model training.

Do not frame AG whole-pass planning as the main LeWM-CEM method unless a separate ground-assisted latent planner is explicitly introduced.

## Still Needed

- Train a non-smoke LeWM checkpoint on the full AUTOPS export.
- Build the LeWM checkpoint + normalizers + probes + utility-weight artifact.
- Run AUTOPS `lewm_cem_eventsat` with `planner_artifact` set to the generated artifact.
- Run paired-seed AO evaluation and refresh the AUTOPS board.
- Measure latency, memory, and rollout throughput on Jetson Orin Nano.

## Future Work

Paper 2 can extend this to CTDE world models for constellation scheduling. Keep that outside the active EventSat repo surface until Paper 1 is stable.
