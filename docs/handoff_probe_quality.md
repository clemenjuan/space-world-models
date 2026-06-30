# Fresh-session task: make the EventSat AO probe stage trustworthy

## Goal
Improve and make *legible* the linear mission-attribute probes that the LeWM-CEM
AO planner depends on. A prior pass misread a raw-unit RMSE and wrongly flagged
`downlink_progress` as broken; the real issues are a **dead** target
(`detection_progress`) and **un-scale-free reporting**. Fix those, keep the
artifact contract intact.

## Repo + environment (everything is local, already set up)
- Repo: `/home/clemente/space-world-models` (work here). Sibling: `/home/clemente/autops-agentic-framework`.
- Python: `/home/clemente/space-world-models/.venv` (uv-made; has torch+hydra+einops+numpy+wandb). Run modules as `.venv/bin/python -m swm_eventsat.experiments.<x>`.
- Trained LeWM checkpoint (real, 150k steps, val_loss 0.130): `outputs/eventsat_autops_lewm/lewm.ckpt` (embed_dim 192, history 3, obs 25, action 7).
- Frozen latents: `outputs/eventsat_autops_latents.npz` (key `latents`, shape (15,10080,192)).
- Dataset: `/home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz` (15×10080, 7D action — current contract; do NOT use any 11D file).
- Probe code: `swm_eventsat/models/probes.py` (`build_attribute_targets`, `fit_ridge_probe`, `ProbeFit`). Probe CLI: `swm_eventsat/experiments/train_autops_probes.py`.

## Corrected diagnosis (verify first, then act)
Run the probe + this per-attribute scale-free check; you should reproduce ~these
`rmse/std` values (judge probes by rmse/std, NOT raw rmse — `fit_ridge_probe`
reports rmse in RAW target units):

| attribute | rmse/std | verdict |
|---|---|---|
| battery_margin | 0.36 | ok |
| storage_margin | 0.24 | ok (tiny variance) |
| downlink_progress | **0.19** | **GOOD (R²≈0.96)** — do not "fix" |
| science_progress | 0.21 | ok |
| detection_progress | **nan (std=0)** | **DEGENERATE** — zero detections in base-EventSat AO traces |
| communication_opportunity | **0.56** | weakest (R²≈0.68) |
| forced_mode_risk | 0.17 | ok |
| anomaly_safe | 0.18 | ok |

## Tasks (in priority order)
1. **Scale-free reporting (required).** Add per-attribute `r2` and `rmse_over_std`
   to `fit_ridge_probe`'s `ProbeFit` and to the `.json` manifest written by
   `train_autops_probes.py`; print a short table. This is the fix that prevents
   the raw-RMSE misread from recurring. Add a `tests/` case asserting a
   high-variance synthetic target gets r2≈1.
2. **Handle degenerate targets (required).** Detect zero-variance attribute
   columns in `build_attribute_targets`/`fit_ridge_probe`, emit a clear warning,
   and report them as degenerate (r2=nan) rather than silent rmse=0. Decide with
   the maintainer whether to (a) keep `detection_progress` in the 8-dim contract
   with utility weight forced to 0, or (b) drop it — see contract constraint below.
3. **Improve `communication_opportunity` (stretch, optional).** It's an
   instantaneous binary ground-pass flag. Try: predicting it from a short latent
   window instead of a single step, or a small 1-hidden-layer head. Only keep a
   change if rmse/std drops meaningfully and it doesn't hurt the others. Do NOT
   turn the linear-probe contract nonlinear without maintainer sign-off (the AO
   backend expects an affine W,b).

## Artifact contract — do NOT break
The AUTOPS AO backend `autops-agentic-framework/src/eventsat/world_model.py`
(`_ArtifactLatentBackend`) reads `probe["W"]` (shape `(n_attributes, embed_dim=192)`),
`probe["b"]`, `probe["attribute_names"]`. `write_planner_artifact.py` asserts
`probe_W.shape[1] == embed_dim`. If you change the attribute set, the utility
presets in `swm_eventsat/planning` (`default_mode_weights`) and the AO backend
must stay consistent — grep both repos for each attribute name before renaming
or dropping. Keep W affine and embed_dim 192.

## Verify end-to-end before finishing
```bash
cd /home/clemente/space-world-models
.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --latents outputs/eventsat_autops_latents.npz \
  --out outputs/eventsat_autops_probe_latent.npz
# new manifest should show r2 / rmse_over_std and flag detection_progress as degenerate
.venv/bin/python -m swm_eventsat.experiments.write_planner_artifact \
  --checkpoint outputs/eventsat_autops_lewm/lewm.ckpt \
  --probe outputs/eventsat_autops_probe_latent.npz \
  --embed-dim 192 --history-size 3 \
  --out outputs/eventsat_autops_lewm/planner_artifact.json   # must still succeed
.venv/bin/python -m pytest tests/ -q   # the probe test you added passes
```

## Out of scope
- Re-training the LeWM (the 150k checkpoint is good).
- Running the AO eval (separate, compute-gated decision).
- The `detection_progress` data gap is a base-EventSat property (detection is an
  SSA concept); don't try to synthesize detections here.

When done: report the new rmse/std + r2 table, what you did about
`detection_progress`, and whether `communication_opportunity` improved.
