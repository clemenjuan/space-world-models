"""Build a local HTML board for FDIR LeWM training and detection results."""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/cache/fdir_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/fdir_results_board.html"
BENCHMARK = ROOT / "data/figures/fdir_detection_benchmark.json"

CHANNELS = [
    "solar_array_voltage",
    "battery_soc",
    "panel_temp",
    "obc_temp",
    "rw_speed_x",
    "rw_speed_y",
    "rw_speed_z",
    "bus_current",
]

SERIES_KEYS = [
    "fit/loss",
    "fit/pred_loss",
    "fit/sigreg_loss",
    "validate/loss_epoch",
    "validate/pred_loss_epoch",
    "validate/sigreg_loss_epoch",
    "hparams/lr_default_0",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt_metric(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.2e}"
    return f"{x:.{digits}g}"


def _extract_metric(sidecar: dict[str, Any], summary: dict[str, Any], key: str) -> float | None:
    value = sidecar.get("summary", {}).get(key)
    if value is not None:
        return _maybe_float(value)
    metric = summary.get("metrics", {}).get(key, {})
    return _maybe_float(metric.get("last"))


def _series_from_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    series = {key: {"x": [], "epoch": [], "y": []} for key in SERIES_KEYS}
    if not path.exists():
        return series
    with path.open(newline="", encoding="utf-8") as fh:
        for row_idx, row in enumerate(csv.DictReader(fh)):
            step = _maybe_float(row.get("step"))
            epoch = _maybe_float(row.get("epoch"))
            x = step if step is not None else float(row_idx)
            for key in SERIES_KEYS:
                y = _maybe_float(row.get(key))
                if y is None:
                    continue
                series[key]["x"].append(x)
                series[key]["epoch"].append(epoch if epoch is not None else float(row_idx))
                series[key]["y"].append(y)
    return series


def _find_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUN_ROOT.exists():
        return runs
    for summary_path in RUN_ROOT.glob("*/*/*/summary.json"):
        run_dir = summary_path.parent
        sidecar = _read_json(run_dir / "sidecar.json")
        summary = _read_json(summary_path)
        hparams = sidecar.get("hparams", {})
        run_id = sidecar.get("run_id") or summary.get("run_id") or run_dir.name
        checkpoint = sidecar.get("checkpoint_path")
        if not checkpoint:
            ckpt = run_dir / "checkpoints/last.ckpt"
            checkpoint = str(ckpt) if ckpt.exists() else ""
        updated = _maybe_float(sidecar.get("updated_at")) or summary_path.stat().st_mtime
        runs.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "name": hparams.get("wandb.config.name") or run_id,
                "status": sidecar.get("status", "unknown"),
                "updated_at": updated,
                "updated_at_iso": datetime.fromtimestamp(updated, timezone.utc).isoformat(),
                "epoch": _extract_metric(sidecar, summary, "epoch"),
                "fit_loss": _extract_metric(sidecar, summary, "fit/loss"),
                "fit_pred_loss": _extract_metric(sidecar, summary, "fit/pred_loss"),
                "fit_sigreg_loss": _extract_metric(sidecar, summary, "fit/sigreg_loss"),
                "val_loss": _extract_metric(sidecar, summary, "validate/loss_epoch")
                or _extract_metric(sidecar, summary, "validate/loss"),
                "val_pred_loss": _extract_metric(sidecar, summary, "validate/pred_loss_epoch")
                or _extract_metric(sidecar, summary, "validate/pred_loss"),
                "val_sigreg_loss": _extract_metric(sidecar, summary, "validate/sigreg_loss_epoch")
                or _extract_metric(sidecar, summary, "validate/sigreg_loss"),
                "lr": _extract_metric(sidecar, summary, "hparams/lr_default_0"),
                "checkpoint": checkpoint,
                "hparams": {
                    "embed_dim": hparams.get("embed_dim"),
                    "history_size": hparams.get("history_size"),
                    "num_preds": hparams.get("num_preds"),
                    "max_epochs": hparams.get("trainer.max_epochs"),
                    "dataset": hparams.get("data.path"),
                    "model_target": hparams.get("model._target_"),
                    "window": hparams.get("data.window"),
                    "batch_size": hparams.get("data.batch_size"),
                    "wandb_entity": hparams.get("wandb.config.entity"),
                    "wandb_project": hparams.get("wandb.config.project"),
                },
                "series": _series_from_csv(run_dir / "metrics.csv"),
            }
        )
    runs.sort(key=lambda r: r["updated_at"], reverse=True)
    return runs


def _is_fdir_run(run: dict[str, Any]) -> bool:
    hparams = run.get("hparams", {})
    return (
        hparams.get("embed_dim") is not None
        and hparams.get("dataset") == "data/cache/fdir_trajectories.npz"
        and hparams.get("model_target") in (None, "models.od_jepa.ODJEPA")
    )


def _fault_rollout(episode_len: int, seed: int = 3) -> dict[str, Any]:
    from envs.fdir_env import FdirEnv

    fault_step = min(100, max(8, episode_len // 2))
    spike_duration = min(20, max(1, episode_len - fault_step))
    env = FdirEnv(
        fault_mode="spike",
        fault_channel="solar_array_voltage",
        fault_step=fault_step,
        spike_magnitude=5.0,
        spike_duration=spike_duration,
        max_steps=episode_len,
    )
    obs, info = env.reset(seed=seed)
    obs_seq = [obs]
    state_seq = [info["state"]]
    active = [bool(info["fault_active"])]
    for _ in range(episode_len - 1):
        obs, _, _, _, info = env.step(0)
        obs_seq.append(obs)
        state_seq.append(info["state"])
        active.append(bool(info["fault_active"]))
    return {
        "obs": np.asarray(obs_seq, dtype=np.float32),
        "state": np.asarray(state_seq, dtype=np.float32),
        "active": np.asarray(active, dtype=bool),
        "fault_step": fault_step,
        "spike_duration": spike_duration,
        "seed": seed,
    }


def _dataset_payload() -> dict[str, Any]:
    if not DATASET.exists():
        return {"ok": False, "message": f"Dataset not found: {DATASET.relative_to(ROOT)}"}

    blob = np.load(DATASET)
    obs = blob["obs"]
    actions = blob["action"]
    state = blob["state"]
    ep = 0
    t = np.arange(obs.shape[1], dtype=float)
    fault = _fault_rollout(obs.shape[1])
    obs_mean = obs.reshape(-1, obs.shape[-1]).mean(axis=0)
    obs_std = obs.reshape(-1, obs.shape[-1]).std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    z_fault = (fault["obs"] - obs_mean) / obs_std
    z_energy = np.mean(np.square(z_fault), axis=1)

    return {
        "ok": True,
        "path": str(DATASET.relative_to(ROOT)),
        "episodes": int(obs.shape[0]),
        "steps": int(obs.shape[1]),
        "obs_dim": int(obs.shape[2]),
        "action_dim": int(actions.shape[2]),
        "state_dim": int(state.shape[2]),
        "sample_episode": ep,
        "time_step": t.tolist(),
        "channels": CHANNELS,
        "nominal": {
            "obs": {name: obs[ep, :, i].astype(float).tolist() for i, name in enumerate(CHANNELS)},
            "state": {name: state[ep, :, i].astype(float).tolist() for i, name in enumerate(CHANNELS)},
        },
        "fault": {
            "obs": {name: fault["obs"][:, i].astype(float).tolist() for i, name in enumerate(CHANNELS)},
            "state": {name: fault["state"][:, i].astype(float).tolist() for i, name in enumerate(CHANNELS)},
            "active": fault["active"].astype(int).tolist(),
            "fault_step": int(fault["fault_step"]),
            "spike_duration": int(fault["spike_duration"]),
            "seed": int(fault["seed"]),
            "z_energy": z_energy.astype(float).tolist(),
        },
        "stats": {
            "obs_mean": {name: float(obs_mean[i]) for i, name in enumerate(CHANNELS)},
            "obs_std": {name: float(obs_std[i]) for i, name in enumerate(CHANNELS)},
            "solar_voltage_mean": float(obs[:, :, 0].mean()),
            "solar_voltage_std": float(obs[:, :, 0].std()),
            "bus_current_mean": float(obs[:, :, 7].mean()),
            "bus_current_std": float(obs[:, :, 7].std()),
        },
    }


def _latest_checkpoint_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for run in runs:
        ckpt = run.get("checkpoint")
        if _is_fdir_run(run) and ckpt and Path(ckpt).exists():
            return run
    return None


def _strip_model_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
    return out or state_dict


def _threshold_stats(values: np.ndarray, fault_step: int, history: int = 0) -> dict[str, float | bool]:
    aligned_fault = max(0, fault_step - history)
    pre_start = max(0, aligned_fault - 20)
    pre = values[pre_start:aligned_fault]
    post = values[aligned_fault:]
    if pre.size == 0 or post.size == 0:
        return {
            "pre_mean": 0.0,
            "pre_std": 0.0,
            "post_mean": 0.0,
            "threshold": 0.0,
            "margin": 0.0,
            "passes": False,
        }
    pre_mean = float(pre.mean())
    pre_std = float(pre.std())
    post_mean = float(post.mean())
    threshold = 3.0 * pre_std
    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "post_mean": post_mean,
        "threshold": threshold,
        "margin": post_mean - pre_mean,
        "passes": bool(post_mean > pre_mean + threshold),
    }


def _model_probe(runs: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    run = _latest_checkpoint_run(runs)
    if run is None:
        return {"ok": False, "message": "No readable FDIR checkpoint found under stable-pretraining runs."}
    if not dataset.get("ok"):
        return {"ok": False, "message": "Dataset unavailable; checkpoint probe skipped."}

    try:
        import hydra
        import torch
        from omegaconf import OmegaConf

        from models.surprise import surprise_score
    except Exception as exc:
        return {"ok": False, "message": f"Probe dependencies unavailable: {exc}"}

    try:
        run_dir = Path(run["run_dir"])
        cfg = OmegaConf.load(run_dir / "hparams.yaml")
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        state_dict = _strip_model_prefix(checkpoint.get("state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        blob = np.load(DATASET)
        obs = blob["obs"]
        actions = blob["action"]
        ep = int(dataset["sample_episode"])
        history = int(cfg.history_size)
        num_preds = int(cfg.num_preds)
        window = int(cfg.data.window)
        fault_step = int(dataset["fault"]["fault_step"])

        obs_mean = obs.reshape(-1, obs.shape[-1]).mean(axis=0)
        obs_std = obs.reshape(-1, obs.shape[-1]).std(axis=0)
        obs_std[obs_std < 1e-8] = 1.0
        act_mean = actions.reshape(-1, actions.shape[-1]).mean(axis=0)
        act_std = actions.reshape(-1, actions.shape[-1]).std(axis=0)
        act_std[act_std < 1e-8] = 1.0

        obs_norm = (obs[ep] - obs_mean) / obs_std
        act_norm = (actions[ep] - act_mean) / act_std
        starts = np.arange(0, obs_norm.shape[0] - window + 1)
        obs_windows = np.stack([obs_norm[s : s + window] for s in starts])
        act_windows = np.stack([act_norm[s : s + window] for s in starts])

        with torch.no_grad():
            batch = {
                "obs": torch.tensor(obs_windows, dtype=torch.float32),
                "action": torch.tensor(act_windows, dtype=torch.float32),
            }
            encoded = model.encode(batch)
            emb = encoded["emb"]
            act_emb = encoded["act_emb"]
            pred = model.predict(emb[:, :history], act_emb[:, :history])
            target = emb[:, num_preds:]
            m = min(pred.size(1), target.size(1))
            pred_last = pred[:, m - 1]
            target_last = target[:, m - 1]
            persistence = emb[:, history - 1]
            mse = (pred_last - target_last).pow(2).mean(dim=1).cpu().numpy()
            persistence_mse = (persistence - target_last).pow(2).mean(dim=1).cpu().numpy()

        fault_obs = np.array([dataset["fault"]["obs"][name] for name in CHANNELS], dtype=np.float32).T
        fault_action = np.zeros((fault_obs.shape[0], actions.shape[-1]), dtype=np.float32)
        fault_action[:, 0] = 1.0
        fault_obs_norm = (fault_obs - obs_mean) / obs_std
        fault_act_norm = (fault_action - act_mean) / act_std
        nominal_episode = torch.tensor(obs_norm[None], dtype=torch.float32)
        nominal_action = torch.tensor(act_norm[None], dtype=torch.float32)
        fault_episode = torch.tensor(fault_obs_norm[None], dtype=torch.float32)
        fault_action_t = torch.tensor(fault_act_norm[None], dtype=torch.float32)

        nominal_scores = surprise_score(model, nominal_episode, nominal_action, history_size=history).cpu().numpy()
        fault_scores = surprise_score(model, fault_episode, fault_action_t, history_size=history).cpu().numpy()
        score_time = np.arange(history, fault_obs.shape[0])
        z_energy = np.asarray(dataset["fault"]["z_energy"], dtype=float)
        z_stats = _threshold_stats(z_energy, fault_step, history=0)
        lewm_stats = _threshold_stats(fault_scores, fault_step, history=history)

        return {
            "ok": True,
            "run_id": run["run_id"],
            "run_name": run["name"],
            "checkpoint": run["checkpoint"],
            "history_size": history,
            "num_preds": num_preds,
            "window": window,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "time_step": score_time.astype(int).tolist(),
            "nominal_surprise": nominal_scores.astype(float).tolist(),
            "fault_surprise": fault_scores.astype(float).tolist(),
            "nominal_mse": mse.astype(float).tolist(),
            "persistence_mse": persistence_mse.astype(float).tolist(),
            "mse_time_step": (starts + num_preds + m - 1).astype(int).tolist(),
            "stats": {
                "nominal_mse_mean": float(mse.mean()),
                "persistence_mse_mean": float(persistence_mse.mean()),
                "nominal_surprise_mean": float(nominal_scores.mean()),
                "fault_surprise_mean": float(fault_scores.mean()),
                "lewm_threshold": lewm_stats,
                "zscore_threshold": z_stats,
            },
        }
    except Exception as exc:
        return {"ok": False, "message": f"Checkpoint probe failed: {exc}"}


def _run_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": run["name"],
            "run_id": run["run_id"],
            "status": run["status"],
            "updated_at_iso": run["updated_at_iso"],
            "epoch": run["epoch"],
            "fit_pred_loss": run["fit_pred_loss"],
            "val_pred_loss": run["val_pred_loss"],
            "fit_loss": run["fit_loss"],
            "val_loss": run["val_loss"],
            "checkpoint": bool(run.get("checkpoint") and Path(run["checkpoint"]).exists()),
            "embed_dim": run["hparams"].get("embed_dim"),
            "history_size": run["hparams"].get("history_size"),
            "max_epochs": run["hparams"].get("max_epochs"),
            "batch_size": run["hparams"].get("batch_size"),
        }
        for run in runs
    ]


def _benchmark_payload() -> dict[str, Any]:
    if not BENCHMARK.exists():
        return {
            "ok": False,
            "message": (
                f"Benchmark artifact not found: {BENCHMARK.relative_to(ROOT)}. "
                "Run .venv/bin/python scripts/evaluate_fdir_detection.py to generate it."
            ),
        }
    payload = _read_json(BENCHMARK)
    if not payload:
        return {"ok": False, "message": f"Could not parse {BENCHMARK.relative_to(ROOT)}."}
    payload = dict(payload)
    payload["ok"] = True
    payload["path"] = str(BENCHMARK.relative_to(ROOT))
    return payload


def build_payload() -> dict[str, Any]:
    all_runs = _find_runs()
    runs = [run for run in all_runs if _is_fdir_run(run)]
    latest = next((r for r in runs if r["series"].get("fit/pred_loss", {}).get("y")), None)
    if latest is None:
        latest = runs[0] if runs else None
    dataset = _dataset_payload()
    probe = _model_probe(runs, dataset)
    benchmark = _benchmark_payload()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "runs": runs,
        "run_table": _run_table(runs),
        "latest_run_id": latest["run_id"] if latest else None,
        "dataset": dataset,
        "probe": probe,
        "benchmark": benchmark,
        "formatted": {
            "latest_val_pred": _fmt_metric(latest["val_pred_loss"] if latest else None),
            "latest_fit_pred": _fmt_metric(latest["fit_pred_loss"] if latest else None),
            "latest_val_loss": _fmt_metric(latest["val_loss"] if latest else None),
            "lewm_margin": _fmt_metric(
                probe.get("stats", {}).get("lewm_threshold", {}).get("margin") if probe.get("ok") else None
            ),
        },
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")
    runs = len(payload["runs"])
    probe = "with checkpoint probe" if payload["probe"].get("ok") else "without checkpoint probe"
    print(f"wrote {OUT.relative_to(ROOT)} from {runs} FDIR run(s), {probe}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Space World Models - FDIR LeWM Board</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {
  --ink: #172026;
  --muted: #5d6872;
  --line: #d8e0e6;
  --panel: #f7f9fb;
  --blue: #1264a3;
  --green: #278a63;
  --gold: #a97000;
  --red: #b43d3d;
  --violet: #6b5aa8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: #ffffff;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
header {
  padding: 24px 44px 18px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #fbfcfd 0%, #ffffff 100%);
}
h1 { margin: 0; font-size: 25px; line-height: 1.15; letter-spacing: 0; color: var(--ink); }
.sub { margin-top: 8px; max-width: 1120px; color: var(--muted); font-size: 13.5px; }
main { padding: 0 44px 34px; }
section { max-width: 1380px; padding: 24px 0 4px; border-bottom: 1px solid #edf1f4; }
section:last-child { border-bottom: 0; }
h2 { margin: 0 0 5px; color: #18364e; font-size: 17px; letter-spacing: 0; }
.caption { margin: 0 0 14px; color: var(--muted); font-size: 12.5px; max-width: 1080px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 10px; margin-top: 18px; }
.kpi { min-height: 82px; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; background: #fff; }
.kpi b { display: block; font-size: 22px; line-height: 1.1; color: var(--blue); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.kpi span { display: block; margin-top: 7px; font-size: 12px; color: var(--muted); }
.gridEven { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; align-items: start; }
.plot { width: 100%; height: 420px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.plot.short { height: 340px; }
.tableWrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
table { border-collapse: collapse; width: 100%; min-width: 760px; font-size: 12.5px; }
th { text-align: left; padding: 9px 10px; color: #314354; background: #f3f6f8; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 8px 10px; border-bottom: 1px solid #edf1f4; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 8px; font-size: 11px; color: var(--muted); white-space: nowrap; }
.pill.ok { color: var(--green); border-color: rgba(39, 138, 99, 0.45); background: rgba(39, 138, 99, 0.08); }
.pill.warn { color: var(--gold); border-color: rgba(169, 112, 0, 0.42); background: rgba(169, 112, 0, 0.08); }
.process { display: grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap: 10px; margin-top: 12px; }
.step { min-height: 128px; border: 1px solid var(--line); border-radius: 8px; padding: 13px; background: #fff; }
.step b { display: block; color: #17344d; font-size: 13px; margin-bottom: 6px; }
.step span { display: block; color: var(--muted); font-size: 12px; }
.note { border-left: 3px solid var(--blue); background: var(--panel); padding: 10px 13px; font-size: 12.5px; color: #334; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }
@media (max-width: 1100px) {
  header, main { padding-left: 22px; padding-right: 22px; }
  .gridEven { grid-template-columns: 1fr; }
  .process { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 650px) {
  header, main { padding-left: 14px; padding-right: 14px; }
  h1 { font-size: 21px; }
  .plot { height: 360px; }
  .process { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>FDIR LeWM Training Board</h1>
  <div class="sub">A local instrument for the fault-detection experiment: nominal telemetry training, state-level fault rollouts, latent surprise detection, calibrated multi-fault benchmarks, and comparison against a simple observation z-score energy baseline. Training curves and the live run are logged to WandB when the trainer runs with W&B enabled.</div>
  <div class="kpis" id="kpis"></div>
</header>
<main>
  <section>
    <h2>Nominal vs Faulted Telemetry</h2>
    <p class="caption">The model trains only on nominal episodes from <code>data/cache/fdir_trajectories.npz</code>. The faulted trace below is a deterministic state-level solar-array-voltage spike rollout used to stress the detector and show coupled telemetry response.</p>
    <div class="gridEven">
      <div id="powerPlot" class="plot"></div>
      <div id="thermalWheelPlot" class="plot"></div>
    </div>
    <div style="height:14px"></div>
    <div class="note" id="datasetNote"></div>
  </section>

  <section>
    <h2>Detection Signal</h2>
    <p class="caption">This is the illustrative single-rollout trace. LeWM surprise is squared latent prediction error; the baseline is raw observation z-score energy under the nominal dataset normalizer. Dashed lines show the local pre-fault relative threshold for this trace.</p>
    <div class="gridEven">
      <div id="surprisePlot" class="plot"></div>
      <div id="comparisonPlot" class="plot"></div>
    </div>
    <div style="height:14px"></div>
    <div class="tableWrap"><table id="detectorTable"></table></div>
    <div style="height:14px"></div>
    <div class="note" id="probeNote"></div>
  </section>

  <section>
    <h2>Fault-Mode Benchmark</h2>
    <p class="caption">Nominal-calibrated thresholds are evaluated across multiple seeds and fault modes. Detection means the score crosses threshold within the configured post-fault window; pre-fault alarms are counted separately.</p>
    <div class="gridEven">
      <div id="benchRatePlot" class="plot short"></div>
      <div id="benchDelayPlot" class="plot short"></div>
    </div>
    <div style="height:14px"></div>
    <div class="tableWrap"><table id="benchTable"></table></div>
    <div style="height:14px"></div>
    <div class="note" id="benchNote"></div>
  </section>

  <section>
    <h2>Training Curves</h2>
    <p class="caption">Newest local FDIR run from stable-pretraining. The run table includes any previous FDIR runs found under <code>~/.cache/stable-pretraining/runs</code>.</p>
    <div class="gridEven">
      <div id="lossPlot" class="plot short"></div>
      <div id="predPlot" class="plot short"></div>
    </div>
    <div style="height:14px"></div>
    <div class="tableWrap"><table id="runTable"></table></div>
  </section>

  <section>
    <h2>Training Process</h2>
    <p class="caption">A compact map of the FDIR path from simulator data to the anomaly score.</p>
    <div class="process">
      <div class="step"><b>Nominal dynamics</b><span>Eight telemetry states evolve around setpoints with stable coupled linear dynamics.</span></div>
      <div class="step"><b>Nominal dataset</b><span>Episodes store noisy observations, true hidden state, and constant nominal one-hot actions.</span></div>
      <div class="step"><b>Encoder</b><span>An MLP maps each 8D telemetry vector into a 192D LeWM embedding.</span></div>
      <div class="step"><b>AR predictor</b><span>A conditional transformer predicts the next latent from history and action embeddings.</span></div>
      <div class="step"><b>Fault rollout</b><span>Faults modify state evolution, then propagate through coupled channels.</span></div>
      <div class="step"><b>Surprise</b><span>High prediction error flags dynamics that are unlikely under the nominal-trained model.</span></div>
    </div>
  </section>
</main>

<script>
const DATA = __PAYLOAD__;
const colors = {blue:"#1264a3", green:"#278a63", gold:"#a97000", red:"#b43d3d", violet:"#6b5aa8", gray:"#6f7a83"};

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "n/a";
  const v = Number(x);
  if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

function pct(x, digits = 0) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "n/a";
  return `${(Number(x) * 100).toFixed(digits)}%`;
}

function esc(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function layout(title, extra = {}) {
  return Object.assign({
    title: {text: title, font: {size: 14, color: "#18364e"}, x: 0.02},
    margin: {t: 48, r: 18, b: 48, l: 58},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: {family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12, color: "#172026"},
    xaxis: {gridcolor: "#edf1f4", zerolinecolor: "#d8e0e6"},
    yaxis: {gridcolor: "#edf1f4", zerolinecolor: "#d8e0e6"},
    legend: {orientation: "h", x: 0, y: -0.18}
  }, extra);
}

function latestRun() {
  return DATA.runs.find(r => r.run_id === DATA.latest_run_id) || DATA.runs[0];
}

function renderKpis() {
  const ds = DATA.dataset || {};
  const probe = DATA.probe || {};
  const bench = DATA.benchmark || {};
  const latest = latestRun();
  const headline = bench.headline || {};
  const cards = [
    [latest ? latest.name : "No FDIR run", "newest local FDIR run"],
    [ds.ok ? `${ds.episodes} x ${ds.steps}` : "n/a", "nominal episodes x steps"],
    [bench.ok ? `${headline.scenario_count} x ${headline.episodes_per_scenario}` : "n/a", "fault scenarios x seeds"],
    [bench.ok ? pct(headline.lewm_mean_detect_rate) : "n/a", "LeWM mean detect rate"],
    [bench.ok ? pct(headline.zscore_mean_detect_rate) : "n/a", "z-score mean detect rate"],
    [bench.ok ? `${fmt(headline.lewm_median_delay, 3)} steps` : "n/a", "LeWM median delay when detected"],
    [DATA.formatted.latest_val_pred, "latest validation pred loss"],
    [probe.ok ? fmt(probe.stats.nominal_mse_mean) : "n/a", "nominal latent MSE"]
  ];
  document.getElementById("kpis").innerHTML = cards.map(([value, label]) =>
    `<div class="kpi"><b>${esc(value)}</b><span>${esc(label)}</span></div>`
  ).join("");
}

function verticalFaultLine(ds) {
  return {
    type: "line",
    x0: ds.fault.fault_step,
    x1: ds.fault.fault_step,
    y0: 0,
    y1: 1,
    yref: "paper",
    line: {color: colors.red, width: 1.5, dash: "dot"}
  };
}

function plotTelemetry() {
  const ds = DATA.dataset || {};
  if (!ds.ok) return;
  const t = ds.time_step;
  Plotly.newPlot("powerPlot", [
    {x: t, y: ds.nominal.obs.solar_array_voltage, type: "scatter", mode: "lines", name: "nominal voltage", line: {color: colors.blue, width: 2}},
    {x: t, y: ds.fault.obs.solar_array_voltage, type: "scatter", mode: "lines", name: "fault voltage", line: {color: colors.red, width: 2}},
    {x: t, y: ds.nominal.obs.battery_soc, type: "scatter", mode: "lines", name: "nominal SoC", yaxis: "y2", line: {color: colors.green, width: 2}},
    {x: t, y: ds.fault.obs.battery_soc, type: "scatter", mode: "lines", name: "fault SoC", yaxis: "y2", line: {color: colors.gold, width: 2}}
  ], layout("Power telemetry", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "voltage V", gridcolor: "#edf1f4"},
    yaxis2: {title: "battery SoC", overlaying: "y", side: "right", showgrid: false},
    shapes: [verticalFaultLine(ds)],
    margin: {t: 48, r: 70, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});

  Plotly.newPlot("thermalWheelPlot", [
    {x: t, y: ds.nominal.obs.panel_temp, type: "scatter", mode: "lines", name: "nominal panel temp", line: {color: colors.blue, width: 2}},
    {x: t, y: ds.fault.obs.panel_temp, type: "scatter", mode: "lines", name: "fault panel temp", line: {color: colors.red, width: 2}},
    {x: t, y: ds.nominal.obs.bus_current, type: "scatter", mode: "lines", name: "nominal bus current", yaxis: "y2", line: {color: colors.green, width: 2}},
    {x: t, y: ds.fault.obs.bus_current, type: "scatter", mode: "lines", name: "fault bus current", yaxis: "y2", line: {color: colors.gold, width: 2}}
  ], layout("Coupled thermal/current telemetry", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "panel temp C", gridcolor: "#edf1f4"},
    yaxis2: {title: "bus current A", overlaying: "y", side: "right", showgrid: false},
    shapes: [verticalFaultLine(ds)],
    margin: {t: 48, r: 76, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});

  document.getElementById("datasetNote").innerHTML =
    `Dataset <code>${esc(ds.path)}</code>: ${ds.episodes} nominal episodes, ${ds.steps} steps, ` +
    `${ds.obs_dim}D observations, ${ds.action_dim}D one-hot actions. Fault rollout seed=${ds.fault.seed}, ` +
    `mode=spike, channel=solar_array_voltage, onset step=${ds.fault.fault_step}, duration=${ds.fault.spike_duration}.`;
}

function plotDetection() {
  const ds = DATA.dataset || {};
  const probe = DATA.probe || {};
  if (!ds.ok) return;
  if (!probe.ok) {
    document.getElementById("probeNote").innerHTML = esc(probe.message || "No checkpoint probe available.");
    Plotly.newPlot("comparisonPlot", [{
      x: ds.time_step,
      y: ds.fault.z_energy,
      type: "scatter",
      mode: "lines",
      name: "obs z-score energy",
      line: {color: colors.gold, width: 2}
    }], layout("Observation baseline before checkpoint probe", {
      xaxis: {title: "step"},
      yaxis: {title: "mean z^2"},
      shapes: [verticalFaultLine(ds)]
    }), {responsive: true, displaylogo: false});
    return;
  }

  const lewmStats = probe.stats.lewm_threshold || {};
  const zStats = probe.stats.zscore_threshold || {};
  const lewmThreshold = Number(lewmStats.pre_mean) + Number(lewmStats.threshold);
  const zThreshold = Number(zStats.pre_mean) + Number(zStats.threshold);
  const lewmThresholdY = Number.isFinite(lewmThreshold) ? Math.max(lewmThreshold, 1e-9) : null;
  const zThresholdY = Number.isFinite(zThreshold) ? Math.max(zThreshold, 1e-9) : null;

  Plotly.newPlot("surprisePlot", [
    {x: probe.time_step, y: probe.nominal_surprise, type: "scatter", mode: "lines", name: "nominal surprise", line: {color: colors.blue, width: 2}},
    {x: probe.time_step, y: probe.fault_surprise, type: "scatter", mode: "lines", name: "fault surprise", line: {color: colors.red, width: 2}},
    {x: probe.time_step, y: probe.time_step.map(() => lewmThresholdY), type: "scatter", mode: "lines", name: "relative threshold", line: {color: colors.gray, width: 1.5, dash: "dash"}}
  ], layout("LeWM latent surprise", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "||z_hat - z||^2", type: "log", gridcolor: "#edf1f4"},
    shapes: [verticalFaultLine(ds)]
  }), {responsive: true, displaylogo: false});

  Plotly.newPlot("comparisonPlot", [
    {x: probe.time_step, y: probe.fault_surprise, type: "scatter", mode: "lines", name: "LeWM surprise", line: {color: colors.violet, width: 2}},
    {x: probe.time_step, y: probe.time_step.map(() => lewmThresholdY), type: "scatter", mode: "lines", name: "LeWM threshold", line: {color: colors.gray, width: 1.2, dash: "dash"}},
    {x: ds.time_step, y: ds.fault.z_energy, type: "scatter", mode: "lines", name: "obs z-score energy", yaxis: "y2", line: {color: colors.gold, width: 2}},
    {x: ds.time_step, y: ds.time_step.map(() => zThresholdY), type: "scatter", mode: "lines", name: "z-score threshold", yaxis: "y2", line: {color: "#9a8f7a", width: 1.2, dash: "dash"}}
  ], layout("Detector score comparison", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "LeWM surprise", type: "log", gridcolor: "#edf1f4"},
    yaxis2: {title: "mean z^2", overlaying: "y", side: "right", type: "log", showgrid: false},
    shapes: [verticalFaultLine(ds)],
    margin: {t: 48, r: 74, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});

  const rows = [
    ["LeWM latent surprise", probe.stats.lewm_threshold],
    ["Observation z-score energy", probe.stats.zscore_threshold]
  ].map(([name, s]) => `<tr>
    <td><b>${esc(name)}</b></td>
    <td class="num">${fmt(s.pre_mean)}</td>
    <td class="num">${fmt(s.pre_std)}</td>
    <td class="num">${fmt(s.post_mean)}</td>
    <td class="num">${fmt(s.threshold)}</td>
    <td class="num">${fmt(s.margin)}</td>
    <td>${s.passes ? '<span class="pill ok">passes</span>' : '<span class="pill warn">below k=3</span>'}</td>
  </tr>`).join("");
  document.getElementById("detectorTable").innerHTML = `<thead><tr>
    <th>Detector</th><th class="num">Pre mean</th><th class="num">Pre std</th><th class="num">Post mean</th>
    <th class="num">3 * std</th><th class="num">Post-pre</th><th>Relative check</th>
  </tr></thead><tbody>${rows}</tbody>`;

  document.getElementById("probeNote").innerHTML =
    `Checkpoint <code>${esc(probe.run_id)}</code> loaded from <code>${esc(probe.checkpoint)}</code>. ` +
    `History=${probe.history_size}, window=${probe.window}, nominal latent MSE=${fmt(probe.stats.nominal_mse_mean)}, ` +
    `persistence baseline=${fmt(probe.stats.persistence_mse_mean)}. Missing keys=${probe.missing_keys}, unexpected keys=${probe.unexpected_keys}.`;
}

function benchmarkCallout(row) {
  if (row.id === "nominal_holdout") return "nominal control";
  const l = Number(row.lewm_detect_rate || 0);
  const z = Number(row.zscore_detect_rate || 0);
  if (l === 0 && z === 0) return "missed by both";
  if (Math.abs(l - z) < 0.05) return l > 0 ? "both detect" : "both quiet";
  return l > z ? "LeWM stronger" : "z-score stronger";
}

function renderBenchmark() {
  const bench = DATA.benchmark || {};
  const table = document.getElementById("benchTable");
  const note = document.getElementById("benchNote");
  if (!table || !note) return;
  const summaries = bench.scenario_summaries || [];
  const faults = summaries.filter(s => s.id !== "nominal_holdout");
  if (!bench.ok || summaries.length === 0) {
    const msg = bench.message || "No FDIR benchmark data available.";
    table.innerHTML = `<tbody><tr><td>${esc(msg)}</td></tr></tbody>`;
    note.innerHTML = esc(msg);
    return;
  }

  const labels = faults.map(s => s.label);
  Plotly.newPlot("benchRatePlot", [
    {x: labels, y: faults.map(s => 100 * Number(s.lewm_detect_rate || 0)), type: "bar", name: "LeWM", marker: {color: colors.violet}, text: faults.map(s => pct(s.lewm_detect_rate)), textposition: "auto"},
    {x: labels, y: faults.map(s => 100 * Number(s.zscore_detect_rate || 0)), type: "bar", name: "z-score", marker: {color: colors.gold}, text: faults.map(s => pct(s.zscore_detect_rate)), textposition: "auto"}
  ], layout("Detection rate within window", {
    barmode: "group",
    xaxis: {tickangle: -18, gridcolor: "#edf1f4"},
    yaxis: {title: "detected episodes %", range: [0, 105], gridcolor: "#edf1f4"},
    margin: {t: 48, r: 18, b: 96, l: 66}
  }), {responsive: true, displaylogo: false});

  const delayValue = (v) => v === null || v === undefined ? 0 : Number(v);
  const delayText = (v) => v === null || v === undefined ? "miss" : `${fmt(v, 3)}`;
  Plotly.newPlot("benchDelayPlot", [
    {x: labels, y: faults.map(s => delayValue(s.lewm_median_delay)), type: "bar", name: "LeWM", marker: {color: colors.violet}, text: faults.map(s => delayText(s.lewm_median_delay)), textposition: "auto"},
    {x: labels, y: faults.map(s => delayValue(s.zscore_median_delay)), type: "bar", name: "z-score", marker: {color: colors.gold}, text: faults.map(s => delayText(s.zscore_median_delay)), textposition: "auto"}
  ], layout("Median detection delay", {
    barmode: "group",
    xaxis: {tickangle: -18, gridcolor: "#edf1f4"},
    yaxis: {title: "steps after fault onset", gridcolor: "#edf1f4"},
    margin: {t: 48, r: 18, b: 96, l: 70}
  }), {responsive: true, displaylogo: false});

  const rows = summaries.map(row => `<tr>
    <td><b>${esc(row.label)}</b><br><span class="pill">${esc(row.fault_mode)} / ${esc(row.fault_channel)}</span></td>
    <td class="num">${pct(row.lewm_detect_rate)}</td>
    <td class="num">${row.lewm_median_delay === null ? "miss" : fmt(row.lewm_median_delay, 3)}</td>
    <td class="num">${pct(row.lewm_false_pre_rate)}</td>
    <td class="num">${fmt(row.lewm_post_pre_ratio, 4)}</td>
    <td class="num">${pct(row.zscore_detect_rate)}</td>
    <td class="num">${row.zscore_median_delay === null ? "miss" : fmt(row.zscore_median_delay, 3)}</td>
    <td class="num">${pct(row.zscore_false_pre_rate)}</td>
    <td class="num">${fmt(row.zscore_post_pre_ratio, 4)}</td>
    <td>${esc(benchmarkCallout(row))}</td>
  </tr>`).join("");
  table.innerHTML = `<thead><tr>
    <th>Scenario</th><th class="num">LeWM detect</th><th class="num">LeWM delay</th><th class="num">LeWM pre alarms</th><th class="num">LeWM lift</th>
    <th class="num">z detect</th><th class="num">z delay</th><th class="num">z pre alarms</th><th class="num">z lift</th><th>Readout</th>
  </tr></thead><tbody>${rows}</tbody>`;

  const cfg = bench.config || {};
  const th = bench.thresholds || {};
  const caveats = (bench.caveats || []).map(c => esc(c)).join("<br>");
  note.innerHTML = `Benchmark <code>${esc(bench.path || "")}</code>, model <code>${esc((bench.model || {}).run_id || "unknown")}</code>, generated ${esc(bench.generated_at || "unknown")}. ` +
    `Thresholds: LeWM ${fmt((th.lewm || {}).threshold)}, z-score ${fmt((th.zscore || {}).threshold)}; ` +
    `${cfg.min_consecutive} consecutive samples, ${cfg.detection_window}-step detection window, ${cfg.seeds} seeds per scenario.<br>${caveats}`;
}

function plotTraining() {
  const latest = latestRun();
  const table = document.getElementById("runTable");
  if (!latest) {
    table.innerHTML = "<tbody><tr><td>No FDIR runs found.</td></tr></tbody>";
    return;
  }
  const s = latest.series || {};
  Plotly.newPlot("lossPlot", [
    {x: (s["fit/loss"] || {}).x || [], y: (s["fit/loss"] || {}).y || [], type: "scatter", mode: "lines+markers", name: "fit loss", line: {color: colors.blue, width: 2}},
    {x: (s["validate/loss_epoch"] || {}).x || [], y: (s["validate/loss_epoch"] || {}).y || [], type: "scatter", mode: "lines+markers", name: "val loss", line: {color: colors.red, width: 2}}
  ], layout("Total loss", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "loss", gridcolor: "#edf1f4"}
  }), {responsive: true, displaylogo: false});
  Plotly.newPlot("predPlot", [
    {x: (s["fit/pred_loss"] || {}).x || [], y: (s["fit/pred_loss"] || {}).y || [], type: "scatter", mode: "lines+markers", name: "fit pred", line: {color: colors.green, width: 2}},
    {x: (s["validate/pred_loss_epoch"] || {}).x || [], y: (s["validate/pred_loss_epoch"] || {}).y || [], type: "scatter", mode: "lines+markers", name: "val pred", line: {color: colors.violet, width: 2}}
  ], layout("Prediction loss", {
    xaxis: {title: "step", gridcolor: "#edf1f4"},
    yaxis: {title: "embedding MSE", gridcolor: "#edf1f4"}
  }), {responsive: true, displaylogo: false});

  const rows = DATA.run_table.map(run => `<tr>
    <td><b>${esc(run.name)}</b><br><span class="pill">${esc(run.run_id)}</span></td>
    <td>${esc(run.status)}</td>
    <td>${esc(run.updated_at_iso)}</td>
    <td class="num">${fmt(run.epoch)}</td>
    <td class="num">${fmt(run.fit_pred_loss)}</td>
    <td class="num">${fmt(run.val_pred_loss)}</td>
    <td class="num">${fmt(run.fit_loss)}</td>
    <td class="num">${fmt(run.val_loss)}</td>
    <td>${run.checkpoint ? '<span class="pill ok">yes</span>' : '<span class="pill warn">no</span>'}</td>
  </tr>`).join("");
  table.innerHTML = `<thead><tr>
    <th>Run</th><th>Status</th><th>Updated</th><th class="num">Epoch</th><th class="num">Fit pred</th>
    <th class="num">Val pred</th><th class="num">Fit loss</th><th class="num">Val loss</th><th>Checkpoint</th>
  </tr></thead><tbody>${rows}</tbody>`;
}

renderKpis();
plotTelemetry();
plotDetection();
renderBenchmark();
plotTraining();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
