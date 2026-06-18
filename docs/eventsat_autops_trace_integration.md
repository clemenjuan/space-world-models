# EventSat AUTOPS Trace Integration

This note defines the local handoff needed to use AUTOPS as the high-fidelity
validation source for the EventSat world-model evaluator and later controller.

## Goal

Use AUTOPS to produce action traces and truth outcomes, then reuse the local
EventSat LeWM tooling for:

- fast prediction of proposed operations traces;
- prediction-vs-truth error reports per agent type;
- out-of-distribution checks for toy-trained models;
- fine-tuning on higher-fidelity AUTOPS dynamics;
- later insertion of a LeWM planner/controller as an AUTOPS agent.

The first AUTOPS use should be validation, not debugging the controller.

## Local Trace Schema

Each exported AUTOPS run should become one `.npz` file or one episode inside a
batched `.npz` with these required arrays:

- `obs`: float32, shape `(episode, time, obs_dim)`.
- `action`: float32, shape `(episode, time, 7)`, one-hot over the EventSat modes.
- `state`: float32, shape `(episode, time, state_dim)`, raw diagnostic state.
- `reward`: float32, shape `(episode, time)`.
- `mode`: int64, shape `(episode, time)`, commanded 7-mode index.
- `resolved_mode`: int64, shape `(episode, time)`, actual executed mode index.
- `forced_mode`: float32, shape `(episode, time)`, `1.0` when AUTOPS changed the
  requested mode.

The 7-mode ordering must match the simplified environment:

```text
0 charging
1 communication
2 payload_observe
3 payload_compress
4 payload_detect
5 payload_send
6 safe
```

## Required Metadata

Store metadata either as scalar `.npz` fields or a sidecar JSON:

- `source_repo`: AUTOPS repository URL or local revision.
- `source_commit`: AUTOPS git commit.
- `agent_type`: one of `rule_based`, `schedule_based`, `rl`, `llm`, `hllm`,
  `hybrid`, `human`, or `lewm_mpc`.
- `agent_name`: concrete AUTOPS representation/agent name.
- `scenario_name`: EventSat scenario/config identifier.
- `episode_seed`: simulator seed.
- `step_duration_s`: exported control step duration.
- `state_names`: ordered raw-state field names.
- `mode_names`: ordered 7-mode names.
- `reward_definition`: short name or version for the reward calculation.

## State Mapping

The first export should include at least these local-compatible fields:

- `battery_soc`
- `obc_data_mb`
- `jetson_raw_mb`
- `jetson_compressed_mb`
- `data_downlinked_mb`
- `uncompressed_observations`
- `compression_progress`
- `undetected_observations`
- `detection_progress`
- `total_observation_s`
- `total_detections`
- `current_mode_idx`
- `in_sunlight`
- `ground_pass_active`
- `time_to_next_eclipse_steps`
- `time_to_next_pass_steps`

AUTOPS can append richer fields after these. The local decoder/evaluator should
read names from `state_names` so later fields do not break compatibility.

## Evaluation Stages

1. Export AUTOPS rule-based and schedule-based traces.
2. Run the toy-trained decoder/evaluator against those traces and report
   prediction-vs-truth errors per state field and reward.
3. Add RL, LLM, HLLM, and hybrid traces.
4. Fine-tune the world model and decoder on AUTOPS traces.
5. Compare toy-trained vs AUTOPS-fine-tuned prediction quality by agent type.
6. Add a LeWM-MPC AUTOPS representation/agent after the local controller is
   stable enough to be worth validating.

## Local Consumer Path

The current local evaluator consumes fixed action traces through:

```text
scripts/evaluate_eventsat_action_trace.py
```

For AUTOPS traces, the next implementation step is to add a `--dataset` mode so
the script can read AUTOPS-exported `obs/action/state/reward` directly instead
of replaying the simplified simulator. That will let the same report compare:

```text
AUTOPS truth trajectory
vs
toy-trained LeWM prediction
vs
AUTOPS-fine-tuned LeWM prediction
```

## Acceptance For The First AUTOPS Bridge

- AUTOPS exports at least one rule-based and one schedule-based EventSat trace.
- The local evaluator can load those traces without running the simplified env.
- The report groups error metrics by `agent_type`.
- The board shows AUTOPS prediction-vs-truth errors next to toy-sim errors.
- A fine-tuned model improves over the toy-trained model on AUTOPS held-out
  traces.
