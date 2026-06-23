# Research Tracker: WM-Based Satellite Autonomy
*Last updated: June 23, 2026*

---

## Paper 1 — Latent WM-Based Mission Planning

**Draft title:** Latent World Model-Based Mission Planning for Autonomous Satellite Operations  
**Code focus:** `space-world-models/swm_eventsat/`  
**Env + baselines:** `autops-agentic-framework/` (EventSat case)

### Methodology (core contribution)

- **World model:** LeWorldModel (LeWM, arXiv:2603.19312) as **transition model**  
  \(\hat{\mathbf{z}}_{t+1} = f_\theta(\hat{\mathbf{z}}_t, \mathbf{a}_t)\)  
  JEPA with SIGReg Gaussian latent; ~15M params, trainable on 1 GPU, edge-friendly.

- **Representation:** latent state \(\hat{\mathbf{z}}_t\) where physical quantities are linearly decodable (shown in LeWM via probing).

- **Linear probes:** for each mission attribute \(k\) (battery, thermal, buffer, pointing, …):  
  \(f_k(\mathbf{z}) = \mathbf{w}_k^\top \mathbf{z} + b_k\)  
  trained on **synthetic telemetry** from the AUTOPS/EventSat simulator.

- **Multi-objective utility in latent space:**  
  terminal-only, deployable form  
  \(U(\hat{\mathbf{z}}_{1:H}) = \mathbf{W}\hat{\mathbf{z}}_H + \mathbf{b}\)  
  with \(\mathbf{w} \in \Delta^K\) acting as **mission mode selector** (science / safe / downlink).

- **Planner:** CEM in latent space (latent MPC): sample action sequences, roll out with LeWM, score via \(U\), execute first action, recede horizon.

### Application domain

- **Scenario:** single-satellite EventSat mission in `autops-agentic-framework`.  
- **Task:** multi-objective mission scheduling balancing main metrics:
  - science data utility
  - battery safety
  - thermal safety
  - downlink buffer
  - pointing/slew constraints
  - more from the framework

LeWM+MPC is implemented as an additional planner using the same simulator interface and logging as existing AUTOPS planners.

### Evaluation

- **Evaluation surface:** EventSat case board (same metrics, same logging).
- **Baselines:**
  - Existing AUTOPS planners (rule-based, LLM-agent, RL, etc.).
  - DreamerV3-based MBRL planner (simulation-only baseline).
  - Random shooting MPC (no learned WM).

- **Key metrics:**
  - Mission utility \(U\) per episode (for different mission modes \(\mathbf{w}\)).
  - Constraint violations (battery/thermal/pointing).
  - Planning latency on NVIDIA Jetson Orin Nano.

**Abstract hook:**  
“We propose a latent world model-based planning framework for satellite mission scheduling. Using LeWorldModel as a fast latent-space simulator combined with linear probes and a multi-objective utility, we perform MPC entirely in imagination and validate the approach on an EventSat mission scenario, comparing against existing planners and a DreamerV3-based baseline, with a focus on edge deployment.”

---

## Paper 2 — CTDE World Models for Constellations (Future Work)

**Draft title:** CTDE World Models for Decentralized Multi-Satellite Constellation Scheduling

- Extend Paper 1 from single-sat planning to **multi-sat coordination** under CTDE.
- **Centralized WM** for joint dynamics (physics prior + learned residuals).
- **Per-satellite modules** with factorized latents (DMAWM-style).
- **Execution:** decentralized policies, WM used only in training.

**Benchmark target:** SatBench (satellite coordination MARL benchmark).  
Compare CTDE-WM against model-free MARL and simpler shared-WM baselines.

---

## Key References (short list)

- **LeWM:** JEPA world model, SIGReg, 15M params, edge-suited.  
- **DreamerV3:** RSSM + actor-critic in imagination, SOTA MBRL; RL-AVIST shows use in space.
- **GAWM, DMAWM, M3W, LOGO, COLA:** CTDE/MARL world-model methods for future constellation work.
- **SatBench:** ready-made multi-satellite coordination benchmark.

---

## PhD Storyline (one-liner per step)

1. EUCASS — model-free MARL for constellation ops.  
2. Paper 1 — single-satellite latent WM + MPC beats existing planners and is deployable on edge.  
3. Paper 2 — CTDE WMs for coordinated autonomy at constellation scale.