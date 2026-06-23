# EventSat LeWM Experiment Alignment

This experiment is the first action-conditioned operations toy problem in this
repo. It takes the EventSat mode and pipeline logic from
`clemenjuan/autops-agentic-framework`, but strips away the full AUTOPS
operations-system comparison so the world-model question stays small.

## Question

Can the same vector LeWM/VectorJEPA stack learn nominal spacecraft operations
dynamics from mode-command trajectories?

Compared with the existing tracks:

- OD has rich orbital measurements, but its action is currently a zero
  acceleration placeholder.
- FDIR has coupled telemetry dynamics, but its nominal training action is
  constant.
- EventSat has simple state variables, but the commanded mode directly changes
  future resources and data-pipeline state.

That makes EventSat the first planning-flavoured experiment in the repo: the
model sees `obs_t` plus a one-hot mode command and predicts the next latent
state.

## Simplified Environment

File: `envs/eventsat_env.py`

The environment models one EventSat-like 6U spacecraft with seven commanded
modes:

```text
charging, communication, payload_observe, payload_compress,
payload_detect, payload_send, safe
```

The nominal dynamics are:

- sunlight/eclipse and ground-pass timing are analytic, seeded at reset;
- mode-dependent power updates battery SoC;
- `payload_observe` creates raw Jetson data;
- `payload_compress` converts one raw observation into compressed Jetson data
  after two steps;
- `payload_detect` creates small OBC metadata after five steps;
- `payload_send` moves compressed Jetson data to the OBC;
- `communication` downlinks OBC data only during ground passes;
- invalid mode commands are resolved to `charging` or `safe` when constraints
  require it.

Anomalies, ground-vs-onboard operational paradigms, human planning delay,
multi-satellite coordination, and ADCS settling are intentionally out of scope
for this first nominal toy problem.

## Observation And Action

Observation is a normalized 25D vector mirroring the AUTOPS Gymnasium wrapper:

```text
resource state:
  battery_soc, obc_fill, jetson_raw_fill, jetson_compressed_fill

orbital timing:
  sin(phase), cos(phase), time_to_eclipse, time_to_pass,
  remaining_pass_duration, episode_progress

flags:
  in_sunlight, ground_pass_active, health_nominal

pipeline:
  uncompressed_obs, compression_progress, undetected_obs,
  detection_progress, downlink_utilization

mode:
  current mode one-hot over the seven EventSat modes
```

Action is a 7D one-hot vector of the commanded mode. The generator also saves
the resolved mode for diagnostics, but the LeWM action input is the command.

## Dataset

File: `data/generate_eventsat.py`

The generator saves the same schema used by EventSat:

```text
obs:    (episode, time, 25)
action: (episode, time, 7)
state:  (episode, time, 16)
```

The default rollout policy is a tiny nominal operator with light exploration:

```text
keep battery healthy
downlink during passes
observe when resources are healthy
compress raw data
detect compressed observations
send compressed data to OBC
charge otherwise
```

This is deliberate. A fully random policy would mostly produce invalid or
unproductive commands, while the first world-model dataset should contain
coherent causal mode sequences.

## Training

Files:

- `train_eventsat.py`
- `config/train/eventsat.yaml`
- `config/train/model/eventsat.yaml`
- `config/train/data/eventsat.yaml`

The model is the existing `VectorJEPA` stack with:

```text
encoder input dim = 25
action input dim  = 7
embed dim         = 192
history size      = 3
num preds         = 1
```

The existing `WindowedTrajectoryDataset`, normalizers, `lewm_forward`, and SIGReg
loss are reused unchanged.

## First Metrics To Inspect

The most useful first diagnostics are:

- latent next-step prediction loss on held-out EventSat trajectories;
- multi-step latent rollout stability under scripted mode sequences;
- whether surprise increases for physically inconsistent schedules later;
- eventually, whether a learned model can score candidate mode schedules before
  executing them.
