"""Build a local HTML board for EventSat LeWM mode-dynamics results."""
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

from envs.eventsat_env import MODE_LIST, STATE_NAMES


DATASET = ROOT / "data/cache/eventsat_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/eventsat_results_board.html"
WEEK = ROOT / "data/figures/eventsat_week_inference.json"

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
                },
                "series": _series_from_csv(run_dir / "metrics.csv"),
            }
        )
    runs.sort(key=lambda r: r["updated_at"], reverse=True)
    return runs


def _is_eventsat_run(run: dict[str, Any]) -> bool:
    hparams = run.get("hparams", {})
    model_target = hparams.get("model_target")
    return (
        hparams.get("dataset") == "data/cache/eventsat_trajectories.npz"
        and (model_target is None or model_target == "models.od_jepa.ODJEPA")
    )


def _mode_hist(mode_idx: np.ndarray) -> dict[str, list[Any]]:
    counts = np.bincount(mode_idx.astype(int), minlength=len(MODE_LIST))
    return {"mode": list(MODE_LIST), "count": counts.astype(int).tolist()}


def _dataset_payload() -> dict[str, Any]:
    if not DATASET.exists():
        return {"ok": False, "message": f"Dataset not found: {DATASET.relative_to(ROOT)}"}

    blob = np.load(DATASET)
    obs = blob["obs"]
    action = blob["action"]
    state = blob["state"]
    reward = blob["reward"] if "reward" in blob else np.zeros(obs.shape[:2], dtype=np.float32)
    mode = blob["mode"] if "mode" in blob else np.argmax(action, axis=-1)
    resolved = blob["resolved_mode"] if "resolved_mode" in blob else mode
    forced = blob["forced_mode"] if "forced_mode" in blob else (mode != resolved).astype(np.float32)
    ep = 0
    s = {name: state[ep, :, i] for i, name in enumerate(STATE_NAMES)}
    t_min = np.arange(obs.shape[1], dtype=float)
    data_stored = s["obc_data_mb"] + s["jetson_raw_mb"] + s["jetson_compressed_mb"]
    modes = mode[ep].astype(int)
    resolved_modes = resolved[ep].astype(int)

    return {
        "ok": True,
        "path": str(DATASET.relative_to(ROOT)),
        "episodes": int(obs.shape[0]),
        "steps": int(obs.shape[1]),
        "obs_dim": int(obs.shape[2]),
        "action_dim": int(action.shape[2]),
        "state_dim": int(state.shape[2]),
        "sample_episode": ep,
        "time_min": t_min.tolist(),
        "mode_names": list(MODE_LIST),
        "state_names": list(STATE_NAMES),
        "soc": s["battery_soc"].tolist(),
        "obc_mb": s["obc_data_mb"].tolist(),
        "raw_mb": s["jetson_raw_mb"].tolist(),
        "compressed_mb": s["jetson_compressed_mb"].tolist(),
        "stored_mb": data_stored.tolist(),
        "downlinked_mb": s["data_downlinked_mb"].tolist(),
        "uncompressed_obs": s["uncompressed_observations"].tolist(),
        "undetected_obs": s["undetected_observations"].tolist(),
        "compression_progress": s["compression_progress"].tolist(),
        "detection_progress": s["detection_progress"].tolist(),
        "in_sunlight": s["in_sunlight"].tolist(),
        "ground_pass": s["ground_pass_active"].tolist(),
        "mode": modes.tolist(),
        "mode_label": [MODE_LIST[i] for i in modes],
        "resolved_mode": resolved_modes.tolist(),
        "resolved_label": [MODE_LIST[i] for i in resolved_modes],
        "forced_mode": forced[ep].tolist(),
        "reward": reward[ep].tolist(),
        "cum_reward": np.cumsum(reward[ep]).tolist(),
        "hist": _mode_hist(mode.reshape(-1)),
        "summary": {
            "final_soc": float(s["battery_soc"][-1]),
            "final_stored_mb": float(data_stored[-1]),
            "final_downlinked_mb": float(s["data_downlinked_mb"][-1]),
            "observation_min": float(s["total_observation_s"][-1] / 60.0),
            "detections": int(s["total_detections"][-1]),
            "forced_rate": float(forced.mean()),
        },
    }


def _week_payload() -> dict[str, Any]:
    if not WEEK.exists():
        return {
            "ok": False,
            "message": f"Week inference not found: {WEEK.relative_to(ROOT)}",
        }
    payload = _read_json(WEEK)
    if not payload:
        return {"ok": False, "message": f"Could not read {WEEK.relative_to(ROOT)}"}
    return payload


def _card(label: str, value: str, sub: str = "") -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def _html(dataset: dict[str, Any], runs: list[dict[str, Any]], week: dict[str, Any]) -> str:
    latest = runs[0] if runs else {}
    payload = json.dumps({"dataset": dataset, "runs": runs, "week": week}, allow_nan=False)
    ds_cards = ""
    if week.get("ok"):
        summary = week["summary"]
        duration_days = week["steps"] * week.get("step_duration_s", 60.0) / 86400.0
        ds_cards = "\n".join(
            [
                _card("Week Rollout", f'{week["steps"]} min', f'{duration_days:.1f} days, seed {week["seed"]}'),
                _card("Final SoC", f'{summary["final_soc"]:.3f}', "week simulation"),
                _card("Downlinked", f'{summary["final_downlinked_mb"]:.2f} MB', "week simulation"),
                _card("Obs Time", f'{summary["observation_min"]:.1f} min', "week simulation"),
                _card("Detections", _fmt_metric(summary["detections"]), "week simulation"),
                _card("Forced Modes", f'{summary["forced_rate"] * 100:.1f}%', "week simulation"),
            ]
        )
    elif dataset.get("ok"):
        summary = dataset["summary"]
        ds_cards = "\n".join(
            [
                _card("Dataset", f'{dataset["episodes"]} x {dataset["steps"]}', dataset["path"]),
                _card("Obs / Action", f'{dataset["obs_dim"]}D / {dataset["action_dim"]}D', "EventSat vector stream"),
                _card("Final SoC", f'{summary["final_soc"]:.3f}', "sample episode"),
                _card("Downlinked", f'{summary["final_downlinked_mb"]:.2f} MB', "sample episode"),
                _card("Obs Time", f'{summary["observation_min"]:.1f} min', "sample episode"),
                _card("Forced Modes", f'{summary["forced_rate"] * 100:.1f}%', "all episodes"),
            ]
        )
    else:
        ds_cards = _card("Dataset", "missing", dataset.get("message", ""))

    week_metrics = week.get("metrics", {}) if week.get("ok") else {}
    week_mse = _maybe_float(week_metrics.get("mse_mean"))
    week_persist = _maybe_float(week_metrics.get("persistence_mse_mean"))
    week_ratio = _maybe_float(week_metrics.get("model_over_persistence_mean"))
    if week_ratio is None and week_mse is not None and week_persist is not None:
        week_ratio = week_mse / max(week_persist, 1e-12)

    run_cards = "\n".join(
        [
            _card("Selected Run", str(latest.get("run_id", "none")), latest.get("status", "waiting")),
            _card("Epoch", _fmt_metric(latest.get("epoch")), latest.get("updated_at_iso", "")),
            _card("Val Pred", _fmt_metric(latest.get("val_pred_loss")), "validate/pred_loss_epoch"),
            _card("Fit Pred", _fmt_metric(latest.get("fit_pred_loss")), "fit/pred_loss"),
            _card("Week MSE", _fmt_metric(week_mse), "one-step latent"),
            _card("Model / Persist", _fmt_metric(week_ratio), "lower is better"),
        ]
    )

    explain_panel = ""
    if week.get("ok"):
        env = week.get("environment", {})
        acct = week.get("data_accounting", {})
        rules = "".join(f"<li>{rule}</li>" for rule in week.get("action_rules", []))
        explain_panel = f"""
  <section class="panel wide explain">
    <h2>Action Source And Learning Target</h2>
    <div class="explain-grid">
      <div>
        <div class="label">Actions</div>
        <p>{week.get("action_source", "n/a")}.</p>
        <ol class="rules">{rules}</ol>
      </div>
      <div>
        <div class="label">Actual Learning</div>
        <p>{week.get("learning_target", "n/a")}.</p>
        <p>The model does not choose downlink, observe, or charge. It learns how the normalized EventSat state changes when a supplied action is taken.</p>
      </div>
      <div>
        <div class="label">OBC / Data Accounting</div>
        <p>OBC capacity is {_fmt_metric(_maybe_float(env.get("obc_capacity_mb")))} MB, not around 200 MB.</p>
        <p>Final OBC/stored data is {_fmt_metric(_maybe_float(acct.get("final_obc_mb")))} MB because generated data minus downlinked data leaves a residual.</p>
        <p>Generated estimate: {_fmt_metric(_maybe_float(acct.get("generated_to_obc_mb_est")))} MB; downlinked: {_fmt_metric(_maybe_float(acct.get("downlinked_mb")))} MB; downlink per communication step: {_fmt_metric(_maybe_float(env.get("downlink_capacity_mb_per_step")))} MB.</p>
      </div>
    </div>
  </section>"""

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EventSat LeWM Board</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ --blue:#0065bd; --green:#2f7d32; --red:#c62828; --gold:#a66a00; --violet:#6a3d9a; --ink:#202124; --muted:#667085; --line:#d9dee7; --bg:#f6f8fb; }}
    body {{ margin:0; font-family:Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:28px 34px 18px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:30px; letter-spacing:0; }}
    .sub {{ color:var(--muted); font-size:13px; line-height:1.45; }}
    main {{ padding:22px 34px 38px; max-width:1380px; margin:0 auto; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:12px 0 22px; }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px; min-height:78px; }}
    .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .value {{ font-size:23px; margin-top:6px; font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .panel {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; min-height:360px; }}
    .wide {{ grid-column:1/-1; }}
    h2 {{ margin:2px 4px 8px; font-size:16px; }}
    .plot {{ width:100%; height:315px; }}
    .explain {{ min-height:auto; }}
    .explain-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; }}
    .explain p {{ margin:8px 0 0; font-size:13px; line-height:1.45; color:var(--ink); }}
    .rules {{ margin:8px 0 0; padding-left:20px; font-size:12px; line-height:1.45; }}
    .rules li {{ margin:3px 0; }}
     (max-width: 980px) {{ .explain-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:18px; }} header {{ padding:22px 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>EventSat LeWM Board</h1>
  <div class="sub">One-week nominal EventSat rollout: resources, contact windows, payload pipeline, commanded modes, and LeWM prediction error against persistence.</div>
</header>
<main>
  <section class="cards">{ds_cards}</section>
  <section class="cards">{run_cards}</section>
  {explain_panel}
  <section class="grid">
    <div class="panel wide"><h2>Week Resource And Contact Timeline</h2><div id="resources" class="plot"></div></div>
    <div class="panel"><h2>Week Data Pipeline</h2><div id="pipeline" class="plot"></div></div>
    <div class="panel"><h2>Week Mode Timeline</h2><div id="modes" class="plot"></div></div>
    <div class="panel"><h2>Week Pipeline Counters</h2><div id="counters" class="plot"></div></div>
    <div class="panel"><h2>Week Action Mix</h2><div id="hist" class="plot"></div></div>
    <div class="panel wide"><h2>Training Curves</h2><div id="training" class="plot"></div></div>
    <div class="panel wide"><h2>Week Prediction Error</h2><div id="week" class="plot"></div></div>
  </section>
</main>
<script>
const DATA = {payload};
const colors = {{blue:"#0065bd", green:"#2f7d32", red:"#c62828", gold:"#a66a00", violet:"#6a3d9a", gray:"#7a869a"}};
const ds = DATA.dataset;
const run = (DATA.runs || [])[0] || null;
const wk = DATA.week || {{ok:false, message:"No week inference"}};
const layoutBase = {{margin:{{t:18,b:48,l:58,r:58}}, paper_bgcolor:"#fff", plot_bgcolor:"#fff", legend:{{orientation:"h", y:-0.22}}, font:{{family:"Arial, sans-serif", size:12}}}};
function plotEmpty(id, msg) {{
  Plotly.newPlot(id, [{{x:[0], y:[0], mode:"text", text:[msg], textposition:"middle center", showlegend:false}}], {{...layoutBase, xaxis:{{visible:false}}, yaxis:{{visible:false}}}}, {{displayModeBar:false, responsive:true}});
}}
if (!wk.ok && !ds.ok) {{
  ["resources","pipeline","modes","counters","hist","training","week"].forEach(id => plotEmpty(id, wk.message || ds.message || "No data"));
}} else {{
  const op = wk.ok ? wk : ds;
  const t = wk.ok ? (wk.time_hour || wk.time_step) : ds.time_min;
  const xTitle = wk.ok ? "hour of week" : "step";
  const modeNames = op.mode_names || ds.mode_names || [];
  const modeLabel = op.mode_label || [];
  const resolvedLabel = op.resolved_label || [];

  Plotly.newPlot("resources", [
    {{x:t, y:op.soc || [], name:"battery SoC", type:"scatter", mode:"lines", line:{{color:colors.blue, width:2}}}},
    {{x:t, y:op.ground_pass || [], name:"ground pass", yaxis:"y2", type:"scatter", mode:"lines", line:{{color:colors.green, width:2, shape:"hv"}}}},
    {{x:t, y:op.in_sunlight || [], name:"sunlight", yaxis:"y2", type:"scatter", mode:"lines", line:{{color:colors.gold, width:2, dash:"dot", shape:"hv"}}}},
    {{x:t, y:op.stored_mb || [], name:"stored MB", yaxis:"y3", type:"scatter", mode:"lines", line:{{color:colors.violet, width:2}}}}
  ], {{...layoutBase, xaxis:{{title:xTitle}}, yaxis:{{title:"SoC", range:[0,1]}}, yaxis2:{{title:"flags", overlaying:"y", side:"right", range:[-0.05,1.05]}}, yaxis3:{{title:"MB", overlaying:"y", side:"right", anchor:"free", position:0.96}}}}, {{responsive:true}});

  Plotly.newPlot("pipeline", [
    {{x:t, y:op.raw_mb || [], name:"Jetson raw", type:"scatter", mode:"lines", line:{{color:colors.red, width:2}}}},
    {{x:t, y:op.compressed_mb || [], name:"Jetson compressed", type:"scatter", mode:"lines", line:{{color:colors.gold, width:2}}}},
    {{x:t, y:op.obc_mb || [], name:"OBC", type:"scatter", mode:"lines", line:{{color:colors.blue, width:2}}}},
    {{x:t, y:op.downlinked_mb || [], name:"downlinked", type:"scatter", mode:"lines", line:{{color:colors.green, width:2}}}}
  ], {{...layoutBase, xaxis:{{title:xTitle}}, yaxis:{{title:"MB"}}}}, {{responsive:true}});

  Plotly.newPlot("modes", [
    {{x:t, y:op.mode || [], text:modeLabel, name:"commanded", type:"scatter", mode:"markers", marker:{{color:op.mode || [], colorscale:"Viridis", size:7}}, hovertemplate:"time=%{{x}}<br>%{{text}}<extra></extra>"}},
    {{x:t, y:op.resolved_mode || [], text:resolvedLabel, name:"resolved", type:"scatter", mode:"markers", marker:{{color:colors.red, size:4, symbol:"x"}}, hovertemplate:"time=%{{x}}<br>%{{text}}<extra></extra>"}}
  ], {{...layoutBase, xaxis:{{title:xTitle}}, yaxis:{{title:"mode", tickvals:[0,1,2,3,4,5,6], ticktext:modeNames}}}}, {{responsive:true}});

  Plotly.newPlot("counters", [
    {{x:t, y:op.uncompressed_obs || [], name:"uncompressed obs", type:"scatter", mode:"lines", line:{{color:colors.red, width:2}}}},
    {{x:t, y:op.undetected_obs || [], name:"undetected obs", type:"scatter", mode:"lines", line:{{color:colors.violet, width:2}}}},
    {{x:t, y:op.compression_progress || [], name:"compression progress", type:"scatter", mode:"lines", line:{{color:colors.gold, width:2}}}},
    {{x:t, y:op.detection_progress || [], name:"detection progress", type:"scatter", mode:"lines", line:{{color:colors.green, width:2}}}}
  ], {{...layoutBase, xaxis:{{title:xTitle}}, yaxis:{{title:"count / progress"}}}}, {{responsive:true}});

  Plotly.newPlot("hist", [{{x:(op.hist || {{mode:[]}}).mode, y:(op.hist || {{count:[]}}).count, type:"bar", marker:{{color:colors.blue}}}}], {{...layoutBase, xaxis:{{title:"commanded mode"}}, yaxis:{{title:"count"}}}}, {{responsive:true}});

  if (run && run.series) {{
    const s = run.series;
    Plotly.newPlot("training", [
      {{x:(s["fit/loss"]||{{}}).x || [], y:(s["fit/loss"]||{{}}).y || [], name:"fit loss", type:"scatter", mode:"lines+markers", line:{{color:colors.blue, width:2}}}},
      {{x:(s["validate/loss_epoch"]||{{}}).x || [], y:(s["validate/loss_epoch"]||{{}}).y || [], name:"val loss", type:"scatter", mode:"lines+markers", line:{{color:colors.red, width:2}}}},
      {{x:(s["fit/pred_loss"]||{{}}).x || [], y:(s["fit/pred_loss"]||{{}}).y || [], name:"fit pred", type:"scatter", mode:"lines+markers", line:{{color:colors.green, width:2}}}},
      {{x:(s["validate/pred_loss_epoch"]||{{}}).x || [], y:(s["validate/pred_loss_epoch"]||{{}}).y || [], name:"val pred", type:"scatter", mode:"lines+markers", line:{{color:colors.violet, width:2}}}}
    ], {{...layoutBase, xaxis:{{title:"step"}}, yaxis:{{title:"loss"}}}}, {{responsive:true}});
  }} else {{
    plotEmpty("training", "No matching EventSat run yet");
  }}

  if (wk.ok) {{
    Plotly.newPlot("week", [
      {{x:wk.score_step, y:wk.mse, name:"LeWM one-step MSE", type:"scatter", mode:"lines", line:{{color:colors.blue, width:2}}}},
      {{x:wk.score_step, y:wk.persistence_mse, name:"persistence MSE", type:"scatter", mode:"lines", line:{{color:colors.red, width:2, dash:"dot"}}}},
      {{x:wk.time_step, y:wk.soc, name:"SoC", yaxis:"y2", type:"scatter", mode:"lines", line:{{color:colors.green, width:1.5}}}},
      {{x:wk.time_step, y:wk.stored_mb, name:"stored MB", yaxis:"y3", type:"scatter", mode:"lines", line:{{color:colors.violet, width:1.5}}}}
    ], {{...layoutBase, xaxis:{{title:"minute"}}, yaxis:{{title:"latent MSE"}}, yaxis2:{{title:"SoC", overlaying:"y", side:"right", range:[0,1]}}, yaxis3:{{title:"MB", overlaying:"y", side:"right", anchor:"free", position:0.96}}}}, {{responsive:true}});
  }} else {{
    plotEmpty("week", wk.message || "No week inference yet");
  }}
}}
</script>
</body>
</html>"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_payload()
    week = _week_payload()
    runs = [run for run in _find_runs() if _is_eventsat_run(run)]
    runs.sort(
        key=lambda run: (
            run.get("val_pred_loss") is None,
            run.get("val_pred_loss") if run.get("val_pred_loss") is not None else float("inf"),
            -(run.get("epoch") or -1),
        )
    )
    OUT.write_text(_html(dataset, runs, week), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
