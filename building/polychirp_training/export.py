from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from building.models import model_eval as M
from building.polychirp_training.results import RunRecord, load_run, run_model_label

METRICS = ("precision", "recall", "f1", "f2", "accuracy", "auc")
SWEEP_METRICS = ("accuracy", "precision", "recall", "f1", "f2")

SCALE_PANELS: list[tuple[str, str, str]] = []
for bk in ("quant", "float"):
    SCALE_PANELS += [
        (f"{bk}_f2_f2", f"Macro F2 ({bk}, F2-tuned)", f"{bk}_macro_f2"),
        (f"{bk}_f1_f1", f"Macro F1 ({bk}, F1-tuned)", f"{bk}_macro_f1"),
        (f"{bk}_f2_precision", f"Macro precision ({bk}, F2-tuned)", f"{bk}_precision_f2"),
        (f"{bk}_f2_recall", f"Macro recall ({bk}, F2-tuned)", f"{bk}_recall_f2"),
        (f"{bk}_f1_precision", f"Macro precision ({bk}, F1-tuned)", f"{bk}_precision_f1"),
        (f"{bk}_f1_recall", f"Macro recall ({bk}, F1-tuned)", f"{bk}_recall_f1"),
        (f"{bk}_f2_accuracy", f"Macro accuracy ({bk}, F2-tuned)", f"{bk}_accuracy"),
        (f"{bk}_f2_auc", f"Macro AUC ({bk})", f"{bk}_auc"),
        (f"{bk}_f2_loss", f"Test loss ({bk})", f"{bk}_loss"),
    ]
SCALE_PANELS += [
    ("drop_f2_f2", "Macro F2 drop (INT8 - float)", "drop_macro_f2"),
    ("drop_f1_f1", "Macro F1 drop (INT8 - float)", "drop_macro_f1"),
    ("drop_f2_precision", "Macro precision drop (INT8 - float)", "drop_precision_f2"),
    ("drop_f2_recall", "Macro recall drop (INT8 - float)", "drop_recall_f2"),
    ("drop_f2_loss", "Test loss rise (INT8 - float)", "drop_loss"),
]


def gather_scale_long(task_dir: Path, keep: set[tuple[str, float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(task_dir.rglob("results.json")):
        try:
            rec = RunRecord.model_validate_json(p.read_text())
        except Exception:
            continue
        tr = rec.train
        extra = rec.model_extra or {}
        if tr is None or extra.get("k") is None or (tr.model.name, tr.model.multi) not in keep:
            continue
        row: dict = {
            "run": run_model_label(tr.model.name, tr.model.multi),
            "build_model": tr.model.name,
            "k": int(extra["k"]),
            "sample_idx": int(extra.get("sample_idx", 0)),
        }
        for backend, bk in (("tflite", "quant"), ("keras", "float")):
            for obj in ("f1", "f2"):
                o = rec.opti_results.get(backend, {}).get("test", {}).get(obj)
                if o is None:
                    continue
                for m in METRICS:
                    row[f"{bk}_{obj}_{m}"] = getattr(o.macro, m)
                row[f"{bk}_{obj}_loss"] = o.loss
        for obj in ("f1", "f2"):
            for m in (*METRICS, "loss"):
                q, f = row.get(f"quant_{obj}_{m}"), row.get(f"float_{obj}_{m}")
                if q is not None and f is not None:
                    row[f"drop_{obj}_{m}"] = q - f
        rows.append(row)
    return pd.DataFrame(rows)


def wide_panel(summary: pd.DataFrame, col: str, runs: list[str]) -> pd.DataFrame:
    ks = sorted({k for _, k in summary.index})
    out = pd.DataFrame({"k": ks})
    for run in runs:
        if run not in summary.index.get_level_values(0):
            continue
        s = summary.loc[run]
        out[run] = [s[(col, "mean")].get(k, np.nan) for k in ks]
        out[f"{run}_sd"] = [s[(col, "std")].fillna(0.0).get(k, np.nan) for k in ks]
    return out


def plot_panel(panel: pd.DataFrame, runs: list[str], ylabel: str, ax) -> None:
    for run in runs:
        if run not in panel.columns:
            continue
        mean, sd = panel[run].to_numpy(), panel[f"{run}_sd"].to_numpy()
        ax.plot(panel["k"], mean, marker="o", label=run)
        ax.fill_between(panel["k"], mean - sd, mean + sd, alpha=0.15)
    ax.set_xlabel("k (target species)")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def export_scaling(
    task_dir: Path,
    keep: set[tuple[str, float]],
    collapse: set[str],
    *,
    panels: list[tuple[str, str, str]] = SCALE_PANELS,
    out_name: str = "final_results",
) -> Path:
    out = task_dir / out_name
    out.mkdir(parents=True, exist_ok=True)
    long = gather_scale_long(task_dir, keep)
    cols = [c for c, _, _ in panels]
    for c in cols:
        if c not in long.columns:
            long[c] = np.nan
    summary = long.groupby(["run", "k"])[cols].agg(["mean", "std"]).sort_index()
    collapse_runs = set(long.loc[long["build_model"].isin(collapse), "run"].unique())

    payload: dict = {"task": task_dir.name, "scaling": {}}
    for col, ylabel, stem in panels:
        runs = sorted(r for r in long["run"].unique()
                      if not (not col.startswith("float_") and r in collapse_runs))
        panel = wide_panel(summary, col, runs)
        panel.to_csv(out / f"scale_{stem}.csv", index=False, float_format="%.5g")
        payload["scaling"][stem] = {
            run: {
                "k": panel["k"].tolist(),
                "mean": panel[run].round(5).tolist(),
                "std": panel[f"{run}_sd"].round(5).tolist(),
            }
            for run in runs if run in panel.columns
        }
        fig, ax = plt.subplots(figsize=(6.3, 4.5))
        plot_panel(panel, runs, ylabel, ax)
        fig.savefig(out / f"scale_{stem}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)
    (out / "scaling.json").write_text(json.dumps(payload, indent=2))
    return out


def export_detection(
    task_dir: Path,
    collection: str,
    models: Iterable,
    collapse: set[str],
    *,
    out_name: str = "final_results",
) -> Path:
    out = task_dir / out_name
    out.mkdir(parents=True, exist_ok=True)
    models = list(models)

    table: dict[str, dict] = {}
    for m in models:
        rp = m.paths(collection).results_path
        if not rp.exists():
            continue
        rec = RunRecord.model_validate_json(rp.read_text())
        entry: dict = {
            "flash_kb": rec.tflite_stats.model_size_kb if rec.tflite_stats else None,
            "collapse": m.base_model.quantize_collapse,
        }
        for backend, bk in (("tflite", "quant"), ("keras", "float")):
            for obj in ("f1", "f2"):
                o = rec.opti_results.get(backend, {}).get("test", {}).get(obj)
                if o is None:
                    continue
                entry[f"{bk}_{obj}"] = {mm: getattr(o.macro, mm) for mm in METRICS}
                entry[f"{bk}_{obj}_loss"] = o.loss
        table[run_model_label(m.name, m.multi)] = entry
    (out / "detection.json").write_text(json.dumps({"task": task_dir.name, "models": table}, indent=2))

    macro = pd.DataFrame([
        {
            "model": lab, "flash_kb": e.get("flash_kb"),
            "f2": e.get("quant_f2", {}).get("f2"), "f1": e.get("quant_f1", {}).get("f1"),
            "precision_f2": e.get("quant_f2", {}).get("precision"),
            "recall_f2": e.get("quant_f2", {}).get("recall"),
            "auc": e.get("quant_f2", {}).get("auc"),
        }
        for lab, e in table.items() if not e.get("collapse")
    ])
    macro.to_csv(out / "det_macro.csv", index=False, float_format="%.5g")

    loss = pd.DataFrame([
        {"model": lab, "loss_float": e.get("float_f2_loss"), "loss_quant": e.get("quant_f2_loss")}
        for lab, e in table.items()
        if e.get("float_f2_loss") is not None and e.get("quant_f2_loss") is not None
    ]).sort_values("loss_quant")
    loss.to_csv(out / "det_loss_drop.csv", index=False, float_format="%.5g")
    x = np.arange(len(loss))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - 0.2, loss["loss_float"], 0.4, label="float", color="#3b6fa3")
    ax.bar(x + 0.2, loss["loss_quant"], 0.4, label="INT8", color="#d97a3f")
    ax.set_xticks(x)
    ax.set_xticklabels(loss["model"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Test loss (BCE)")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.savefig(out / "det_loss_drop.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    for obj in ("f2", "f1"):
        d = macro.sort_values(obj, ascending=False)
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        ax.bar(np.arange(len(d)), d[obj], color="#3b6fa3")
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["model"], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(f"macro {obj.upper()} (INT8)")
        ax.grid(True, axis="y", ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / f"det_bars_{obj}.pdf")
        plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(macro["flash_kb"], macro["f2"], color="#d97a3f")
    for _, r in macro.iterrows():
        ax.annotate(r["model"], (r["flash_kb"], r["f2"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("flash (KB)")
    ax.set_ylabel("macro F2 (INT8)")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "det_flash_vs_f2.pdf")
    plt.show()
    plt.close(fig)

    evals = {
        run_model_label(m.name, m.multi): load_run(m.paths(collection).results_path).eval("tflite", "test", "f2")
        for m in models
        if not m.base_model.quantize_collapse and m.paths(collection).results_path.exists()
    }
    thr = np.linspace(0.0, 1.0, 101)
    for metric in SWEEP_METRICS:
        sweep = pd.DataFrame({"t": thr})
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for label, ev in evals.items():
            curves = [
                M.threshold_curve((ev.y_true[:, c] >= 0.5).astype(int), ev.y_score[:, c], thr)[metric]
                for c in range(len(ev.label_names))
                if (ev.y_true[:, c] >= 0.5).sum()
            ]
            if not curves:
                continue
            mac = np.mean(curves, axis=0)
            sweep[label] = mac
            ax.plot(thr, mac, label=label, lw=1.2)
        sweep.to_csv(out / f"det_sweep_{metric}.csv", index=False, float_format="%.5g")
        ax.set_xlabel("threshold t")
        ax.set_ylabel(f"macro {metric}")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.savefig(out / f"det_sweep_{metric}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)
    return out
