"""Unified evaluation for float Keras models and INT8 TFLite flatbuffers.

Both backends share the same evaluate() surface. EvalResult holds per-class
precision/recall/F1/F2 at a threshold, macro/top-1, a confusion matrix, an
ROC/PR sweep, plus the size/arena/MFLOPs numbers for the baked TFLite version.
QuantComparison drives the quant minus float drop table and bar chart.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import keras
    import matplotlib.figure
    import pandas as pd
else:
    keras = tf.keras

from building.models.bake import TFLiteStats, analyze_tflite

Backend = Literal["keras", "tflite"]
ThresholdMode = Literal["fixed", "best_f1", "best_f2"]


# Prediction backends


def predict_keras(
    model: "keras.Model", ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    t0 = time.perf_counter()
    n = 0
    for batch in ds:
        xb, yb = batch[0], batch[1]
        probs = model(xb, training=False).numpy()
        y_score.append(probs)
        y_true.append(yb.numpy())
        n += probs.shape[0]
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n) * 1000.0 if n else 0.0
    return np.concatenate(y_true), np.concatenate(y_score), avg_ms


def predict_tflite(
    tflite_path: Path, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_dtype = inp["dtype"]
    out_dtype = out["dtype"]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    in_rank = len(inp["shape"])

    # Per-sample quantize/dequantize closures for the converter's I/O dtype:
    # INT8/UINT8 full-integer PTQ, INT16 INT16x8, FLOAT32 weight-only PTQ.
    if in_dtype == np.float32:
        quantize_input = lambda x: x.astype(np.float32)[None, ...]
    elif in_dtype in (np.int8, np.uint8):
        qmin, qmax = (-128, 127) if in_dtype == np.int8 else (0, 255)
        quantize_input = lambda x: np.clip(
            np.round(x / in_scale) + in_zp, qmin, qmax
        ).astype(in_dtype)[None, ...]
    elif in_dtype == np.int16:
        quantize_input = lambda x: np.clip(
            np.round(x / in_scale) + in_zp, -32768, 32767
        ).astype(np.int16)[None, ...]
    else:
        raise ValueError(f"{tflite_path} has unsupported input dtype {in_dtype}")

    if out_dtype == np.float32:
        dequantize_output = lambda raw: raw.astype(np.float32)
    else:
        dequantize_output = lambda raw: (raw.astype(np.float32) - out_zp) * out_scale

    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    times: list[float] = []
    for batch in ds:
        xb, yb = batch[0].numpy(), batch[1].numpy()
        for i in range(xb.shape[0]):
            x = xb[i]
            if x.ndim != in_rank - 1:
                raise ValueError(
                    f"Sample rank {x.ndim} does not match TFLite input rank "
                    f"{in_rank - 1} (input shape {inp['shape']})."
                )
            xq = quantize_input(x)
            interp.set_tensor(inp["index"], xq)
            t0 = time.perf_counter()
            interp.invoke()
            times.append(time.perf_counter() - t0)
            raw = interp.get_tensor(out["index"])
            y_score.append(dequantize_output(raw).reshape(-1))
            y_true.append(yb[i])
    avg_ms = float(np.mean(times) * 1000.0) if times else 0.0
    return np.asarray(y_true), np.asarray(y_score), avg_ms


# Metrics


class PerClassMetrics(BaseModel):
    name: str
    threshold: float
    support: int
    precision: float
    recall: float
    f1: float
    f2: float
    accuracy: float
    auc: float | None = None


class EvalResult(BaseModel):
    """Per-backend evaluation summary and raw scores."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Backend
    label_names: list[str]
    per_class: dict[str, PerClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_f2: float
    macro_accuracy: float
    macro_auc: float | None
    top1_accuracy: float
    subset_accuracy: float  # all-classes-correct rate under each class's threshold
    confusion: np.ndarray  # (K+1, K+1); row/col K is non_target
    avg_inference_ms: float
    threshold_mode: ThresholdMode
    tflite_stats: TFLiteStats | None = None
    y_true: np.ndarray | None = Field(default=None, repr=False)
    y_score: np.ndarray | None = Field(default=None, repr=False)


def threshold_curve(
    y_true_c: np.ndarray,
    y_score_c: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-class metric-vs-threshold sweep: dict of (len(thresholds),) arrays
    for threshold, precision, recall, f1, f2, accuracy."""
    yt = (np.asarray(y_true_c) >= 0.5).astype(int)
    ys = np.asarray(y_score_c)
    thr = np.asarray(thresholds, dtype=float)

    n = yt.size
    precs = np.empty_like(thr)
    recs = np.empty_like(thr)
    f1s = np.empty_like(thr)
    f2s = np.empty_like(thr)
    accs = np.empty_like(thr)
    pos = int(yt.sum())
    neg = n - pos
    for i, t in enumerate(thr):
        yp = (ys >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = pos - tp
        tn = neg - fp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        d2 = 4 * p + r
        f2 = 5 * p * r / d2 if d2 else 0.0
        precs[i] = p
        recs[i] = r
        f1s[i] = f1
        f2s[i] = f2
        accs[i] = (tp + tn) / n if n else 0.0
    return {
        "threshold": thr,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
        "f2": f2s,
        "accuracy": accs,
    }


def roc_arrays(
    y_true_c: np.ndarray, y_score_c: np.ndarray
) -> dict[str, np.ndarray | float | None]:
    """Per-class ROC {fpr, tpr, thresholds, auc}; auc None when degenerate."""
    from sklearn.metrics import roc_auc_score, roc_curve

    yt = (np.asarray(y_true_c) >= 0.5).astype(int)
    ys = np.asarray(y_score_c)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "thresholds": np.array([np.inf, -np.inf]),
            "auc": None,
        }
    fpr, tpr, thr = roc_curve(yt, ys)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc": float(roc_auc_score(yt, ys)),
    }


def best_threshold(
    y_true_c: np.ndarray, y_score_c: np.ndarray, beta: float
) -> tuple[float, float]:
    """Return (threshold, Fbeta) maximising Fbeta on this class's PR curve."""
    from sklearn.metrics import precision_recall_curve

    if y_true_c.sum() == 0:
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(y_true_c, y_score_c)
    prec, rec = prec[:-1], rec[:-1]
    if thr.size == 0:
        return 0.5, 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    fb = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
    i = int(np.argmax(fb))
    return float(thr[i]), float(fb[i])


def per_class_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
) -> tuple[dict[str, PerClassMetrics], float | None]:
    """Per-class precision/recall/F1/F2/AUC + macro AUC."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, PerClassMetrics] = {}
    aucs: list[float] = []
    for c, name in enumerate(label_names):
        yt = (y_true[:, c] >= 0.5).astype(int)
        ys = y_score[:, c]
        thr = float(thresholds[c])
        yp = (ys >= thr).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        denom2 = 4 * prec + rec
        f2 = 5 * prec * rec / denom2 if denom2 else 0.0
        n = tp + fp + fn + tn
        accuracy = (tp + tn) / n if n else 0.0
        auc: float | None = None
        if yt.sum() and yt.sum() < yt.size:
            auc = float(roc_auc_score(yt, ys))
            aucs.append(auc)
        out[name] = PerClassMetrics(
            name=name,
            threshold=thr,
            support=int(yt.sum()),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            f2=float(f2),
            accuracy=float(accuracy),
            auc=auc,
        )
    macro_auc = float(np.mean(aucs)) if aucs else None
    return out, macro_auc


def resolve_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    threshold: float | np.ndarray,
    mode: ThresholdMode,
) -> np.ndarray:
    """Return a per-class threshold vector for `mode`.

    - fixed: broadcast threshold to all classes.
    - best_f1/best_f2: argmax-Fbeta per class on the given y_true/y_score (pass
      the tuning split, not test).
    """
    if mode == "fixed":
        if isinstance(threshold, (int, float)):
            return np.full(len(label_names), float(threshold))
        thr = np.asarray(threshold, dtype=float)
        if thr.shape != (len(label_names),):
            raise ValueError(
                f"threshold array shape {thr.shape} != ({len(label_names)},)"
            )
        return thr

    beta = 1.0 if mode == "best_f1" else 2.0
    thrs = np.empty(len(label_names))
    for c in range(len(label_names)):
        thrs[c], _ = best_threshold(
            (y_true[:, c] >= 0.5).astype(int), y_score[:, c], beta
        )
    return thrs


def top1_with_non_target(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-1 labels and confusion over K+1 buckets (last = non_target).

    True class is argmax(y_true) for rows with any positive label, else K.
    Predicted class is the highest-scoring class above its own threshold, else
    K (non_target).
    """
    k = y_true.shape[1]
    positive = y_true.sum(axis=1) > 0
    yt = np.where(positive, np.argmax(y_true, axis=1), k)

    above = y_score >= thresholds[None, :]
    masked = np.where(above, y_score, -np.inf)
    yp = np.where(above.any(axis=1), np.argmax(masked, axis=1), k)

    cm = np.zeros((k + 1, k + 1), dtype=int)
    np.add.at(cm, (yt, yp), 1)
    return yt, yp, cm


def eval_from_scores(
    *,
    backend: Backend,
    label_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    avg_inference_ms: float,
    threshold_mode: ThresholdMode,
    tflite_stats: TFLiteStats | None = None,
) -> EvalResult:
    """Wrap pre-computed predictions in an EvalResult."""
    per_class, macro_auc = per_class_metrics(
        y_true, y_score, label_names, thresholds
    )

    precisions = [m.precision for m in per_class.values()]
    recalls = [m.recall for m in per_class.values()]
    f1s = [m.f1 for m in per_class.values()]
    f2s = [m.f2 for m in per_class.values()]
    accs = [m.accuracy for m in per_class.values()]

    # Top-1/confusion span K+1 buckets: K targets plus non_target at index K,
    # which catches all-zero rows and samples below every threshold.
    yt_top1, yp_top1, cm = top1_with_non_target(y_true, y_score, thresholds)
    top1 = float(np.mean(yt_top1 == yp_top1))

    yt_bin = (y_true >= 0.5).astype(int)
    yp_bin = (y_score >= thresholds[None, :]).astype(int)
    subset = float(np.mean(np.all(yt_bin == yp_bin, axis=1)))

    return EvalResult(
        backend=backend,
        label_names=list(label_names),
        per_class=per_class,
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        macro_f2=float(np.mean(f2s)),
        macro_accuracy=float(np.mean(accs)),
        macro_auc=macro_auc,
        top1_accuracy=top1,
        subset_accuracy=subset,
        confusion=cm,
        avg_inference_ms=avg_inference_ms,
        threshold_mode=threshold_mode,
        tflite_stats=tflite_stats,
        y_true=y_true,
        y_score=y_score,
    )


def predict_via_tflite(
    tflite_path: Path | str, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    """Public wrapper around the int8 inference loop: dequantized probabilities
    without the full evaluate() pipeline."""
    return predict_tflite(Path(tflite_path), ds)


def predict_keras_array(
    model: "keras.Model", x: np.ndarray
) -> tuple[np.ndarray, float]:
    """Batched float inference over a feature array. Returns (y_score, avg_ms)."""
    scores: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(0, len(x), 256):
        scores.append(model(x[i : i + 256], training=False).numpy())
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / len(x)) * 1000.0
    return np.concatenate(scores), avg_ms


def predict_tflite_array(
    tflite_path: Path | str, x: np.ndarray
) -> tuple[np.ndarray, float]:
    """Per-sample INT8 inference over a feature array. Builds its own
    Interpreter, so concurrent calls are thread-safe (invoke() drops the GIL).
    Returns (y_score, avg_ms)."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_dtype, out_dtype = inp["dtype"], out["dtype"]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    in_rank = len(inp["shape"])

    if in_dtype == np.float32:
        quantize_input = lambda s: s.astype(np.float32)[None, ...]
    elif in_dtype in (np.int8, np.uint8):
        qmin, qmax = (-128, 127) if in_dtype == np.int8 else (0, 255)
        quantize_input = lambda s: np.clip(
            np.round(s / in_scale) + in_zp, qmin, qmax
        ).astype(in_dtype)[None, ...]
    elif in_dtype == np.int16:
        quantize_input = lambda s: np.clip(
            np.round(s / in_scale) + in_zp, -32768, 32767
        ).astype(np.int16)[None, ...]
    else:
        raise ValueError(f"{tflite_path} has unsupported input dtype {in_dtype}")

    if out_dtype == np.float32:
        dequantize_output = lambda raw: raw.astype(np.float32)
    else:
        dequantize_output = lambda raw: (raw.astype(np.float32) - out_zp) * out_scale

    scores: list[np.ndarray] = []
    times: list[float] = []
    for s in x:
        if s.ndim != in_rank - 1:
            raise ValueError(
                f"Sample rank {s.ndim} does not match TFLite input rank "
                f"{in_rank - 1} (input shape {inp['shape']})."
            )
        interp.set_tensor(inp["index"], quantize_input(s))
        t0 = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t0)
        scores.append(dequantize_output(interp.get_tensor(out["index"])).reshape(-1))
    avg_ms = float(np.mean(times) * 1000.0) if times else 0.0
    return np.asarray(scores), avg_ms


def pick_best_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    mode: ThresholdMode,
) -> np.ndarray:
    """Per-class argmax-Fbeta thresholds for best_f1/best_f2, swept on the
    given split (pass the tuning split, not test)."""
    if mode == "fixed":
        raise ValueError("pick_best_thresholds is for best_f1/best_f2 only")
    return resolve_thresholds(y_true, y_score, label_names, 0.5, mode)


def eval_with_thresholds(
    *,
    backend: Backend,
    label_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    avg_inference_ms: float = 0.0,
    threshold_mode: ThresholdMode = "fixed",
    tflite_stats: TFLiteStats | None = None,
) -> EvalResult:
    """Build an EvalResult from already-computed predictions + fixed thresholds."""
    return eval_from_scores(
        backend=backend,
        label_names=list(label_names),
        y_true=np.asarray(y_true),
        y_score=np.asarray(y_score),
        thresholds=np.asarray(thresholds, dtype=float),
        avg_inference_ms=avg_inference_ms,
        threshold_mode=threshold_mode,
        tflite_stats=tflite_stats,
    )


def eval_result_from_predictions(
    *,
    backend: Backend,
    label_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    threshold_mode: ThresholdMode = "fixed",
    threshold_tuning: tuple[np.ndarray, np.ndarray] | None = None,
    avg_inference_ms: float = 0.0,
    tflite_stats: TFLiteStats | None = None,
) -> EvalResult:
    """Build an EvalResult from already-computed predictions."""
    if threshold_mode == "fixed":
        thresholds = resolve_thresholds(
            y_true, y_score, label_names, threshold, "fixed"
        )
    else:
        if threshold_tuning is None:
            tune_y_true, tune_y_score = y_true, y_score
        else:
            tune_y_true, tune_y_score = threshold_tuning
        thresholds = resolve_thresholds(
            tune_y_true, tune_y_score, label_names, threshold, threshold_mode
        )
    return eval_from_scores(
        backend=backend,
        label_names=list(label_names),
        y_true=np.asarray(y_true),
        y_score=np.asarray(y_score),
        thresholds=thresholds,
        avg_inference_ms=avg_inference_ms,
        threshold_mode=threshold_mode,
        tflite_stats=tflite_stats,
    )


def evaluate(
    model_or_path: "keras.Model | str | Path",
    test_ds: tf.data.Dataset,
    label_names: list[str],
    *,
    threshold: float | np.ndarray = 0.5,
    threshold_mode: ThresholdMode = "fixed",
    threshold_tuning_ds: tf.data.Dataset | None = None,
) -> EvalResult:
    """Run a model end-to-end and return a fully-populated EvalResult.

    model_or_path is a keras.Model (float) or a path to a .tflite (INT8).
    threshold_mode 'fixed' uses threshold (scalar or per-class array);
    'best_f1'/'best_f2' tune on threshold_tuning_ds if given, else on the test
    set (a peek, beware if you care about strict generalization).
    """
    is_tflite = isinstance(model_or_path, (str, Path))
    backend: Backend = "tflite" if is_tflite else "keras"
    if is_tflite:
        tflite_path = Path(model_or_path)
        y_true, y_score, avg_ms = predict_tflite(tflite_path, test_ds)
        tflite_stats: TFLiteStats | None = analyze_tflite(tflite_path)
    else:
        y_true, y_score, avg_ms = predict_keras(model_or_path, test_ds)
        tflite_stats = None

    if threshold_mode == "fixed":
        thresholds = resolve_thresholds(
            y_true, y_score, label_names, threshold, "fixed"
        )
    else:
        if threshold_tuning_ds is None:
            tune_y_true, tune_y_score = y_true, y_score
        elif is_tflite:
            tune_y_true, tune_y_score, _ = predict_tflite(
                tflite_path, threshold_tuning_ds
            )
        else:
            tune_y_true, tune_y_score, _ = predict_keras(
                model_or_path, threshold_tuning_ds
            )
        thresholds = resolve_thresholds(
            tune_y_true, tune_y_score, label_names, threshold, threshold_mode
        )

    return eval_from_scores(
        backend=backend,
        label_names=list(label_names),
        y_true=y_true,
        y_score=y_score,
        thresholds=thresholds,
        avg_inference_ms=avg_ms,
        threshold_mode=threshold_mode,
        tflite_stats=tflite_stats,
    )


# Reporting helpers


def per_class_table(result: EvalResult) -> "pd.DataFrame":
    import pandas as pd

    rows = [
        {
            "class": m.name,
            "support": m.support,
            "threshold": round(m.threshold, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "f2": round(m.f2, 4),
            "accuracy": round(m.accuracy, 4),
            "auc": None if m.auc is None else round(m.auc, 4),
        }
        for m in result.per_class.values()
    ]
    return pd.DataFrame(rows)


def summary(result: EvalResult) -> dict[str, Any]:
    s: dict[str, Any] = {
        "backend": result.backend,
        "threshold_mode": result.threshold_mode,
        "top1_accuracy": result.top1_accuracy,
        "subset_accuracy": result.subset_accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "macro_f2": result.macro_f2,
        "macro_auc": result.macro_auc,
        "avg_inference_ms": result.avg_inference_ms,
    }
    s["macro_accuracy"] = result.macro_accuracy
    if result.tflite_stats is not None:
        s["model_size_kb"] = result.tflite_stats.model_size_kb
        s["arena_size_kb"] = result.tflite_stats.arena_size_kb
        s["flops_mflops"] = result.tflite_stats.flops_mflops
    return s


def print_summary(result: EvalResult) -> None:
    print(f"=== {result.backend.upper()} evaluation ===")
    print(f"  threshold mode    : {result.threshold_mode}")
    print(f"  top-1 accuracy    : {result.top1_accuracy:.4f}")
    print(f"  subset accuracy   : {result.subset_accuracy:.4f}")
    print(f"  macro precision   : {result.macro_precision:.4f}")
    print(f"  macro recall      : {result.macro_recall:.4f}")
    print(f"  macro F1          : {result.macro_f1:.4f}")
    print(f"  macro F2          : {result.macro_f2:.4f}")
    print(f"  macro accuracy    : {result.macro_accuracy:.4f}")
    if result.macro_auc is not None:
        print(f"  macro AUC         : {result.macro_auc:.4f}")
    print(f"  avg inference (ms): {result.avg_inference_ms:.3f}")
    if result.tflite_stats is not None:
        ts = result.tflite_stats
        print(f"  flash (weights)   : {ts.model_size_kb:.1f} KB")
        if ts.arena_size_kb is not None:
            print(f"  arena (activ.)    : {ts.arena_size_kb:.1f} KB")
        print(f"  est. MFLOPs       : {ts.flops_mflops:.3f}")


# Threshold-selection plots


def plot_roc(
    result: EvalResult,
    classes: Iterable[str] | None = None,
) -> "matplotlib.figure.Figure":
    """One ROC per class, log-x to highlight the low-FPR regime."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    target = list(classes) if classes is not None else list(result.label_names)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, name in enumerate(result.label_names):
        if name not in target:
            continue
        yt = (result.y_true[:, c] >= 0.5).astype(int)
        if yt.sum() == 0 or yt.sum() == yt.size:
            continue
        fpr, tpr, _ = roc_curve(yt, result.y_score[:, c])
        auc = result.per_class[name].auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})" if auc else name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC per class")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_metric_sweep_panel(
    result: EvalResult,
    classes: Iterable[str] | None = None,
    n_thresholds: int = 101,
) -> "matplotlib.figure.Figure":
    """Five-panel figure: accuracy, precision, recall, F1, F2 vs threshold,
    one line per class per panel."""
    import matplotlib.pyplot as plt

    target = list(classes) if classes is not None else list(result.label_names)
    thr = np.linspace(0.0, 1.0, n_thresholds)
    metrics = ("accuracy", "precision", "recall", "f1", "f2")
    fig, axes = plt.subplots(1, 5, figsize=(18,5), sharey=True)
    for c, name in enumerate(result.label_names):
        if name not in target:
            continue
        yt = (result.y_true[:, c] >= 0.5).astype(int)
        if yt.sum() == 0:
            continue
        curve = threshold_curve(yt, result.y_score[:, c], thr)
        for ax, m in zip(axes, metrics):
            ax.plot(thr, curve[m], label=name, linewidth=1.0)
    for ax, m in zip(axes, metrics):
        ax.set_title(m)
        ax.set_xlabel("t")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Per-class metrics vs threshold")
    fig.tight_layout()
    return fig


def plot_model_sweep_panels(
    evals_by_label: dict[str, EvalResult],
    n_thresholds: int = 101,
) -> "matplotlib.figure.Figure":
    """2x3 grid (accuracy, precision, recall, F1, F2 vs a uniform threshold),
    one line per model. Each line is the metric macro-averaged over target
    classes; the sixth axis holds the model legend."""
    import matplotlib.pyplot as plt

    thr = np.linspace(0.0, 1.0, n_thresholds)
    metrics = ("accuracy", "precision", "recall", "f1", "f2")
    fig, axes = plt.subplots(2, 3, figsize=(18, 16), sharey=True)
    axes = axes.flatten()
    for label, result in evals_by_label.items():
        curves = [
            threshold_curve((result.y_true[:, c] >= 0.5).astype(int), result.y_score[:, c], thr)
            for c, name in enumerate(result.label_names)
            if (result.y_true[:, c] >= 0.5).sum()
        ]
        if not curves:
            continue
        macro = {m: np.mean([cv[m] for cv in curves], axis=0) for m in metrics}
        for ax, m in zip(axes, metrics):
            ax.plot(thr, macro[m], label=label, linewidth=1.4)
    for ax, m in zip(axes, metrics):
        ax.set_title(m)
        ax.set_xlabel("t")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("macro score")
    axes[3].set_ylabel("macro score")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", fontsize=10, title="model")
    fig.suptitle("Macro metrics vs threshold, per model")
    fig.tight_layout()
    return fig


# Float vs quantized comparison


class QuantComparison(BaseModel):
    float_eval: EvalResult
    quant_eval: EvalResult


def comparison_table(cmp: QuantComparison) -> "pd.DataFrame":
    """Side-by-side per-class metrics with quant minus float drops."""
    import pandas as pd

    rows = []
    for name in cmp.float_eval.label_names:
        f = cmp.float_eval.per_class[name]
        q = cmp.quant_eval.per_class[name]
        rows.append(
            {
                "class": name,
                "support": f.support,
                "f1_float": round(f.f1, 4),
                "f1_int8": round(q.f1, 4),
                "df1": round(q.f1 - f.f1, 4),
                "f2_float": round(f.f2, 4),
                "f2_int8": round(q.f2, 4),
                "df2": round(q.f2 - f.f2, 4),
                "prec_float": round(f.precision, 4),
                "prec_int8": round(q.precision, 4),
                "rec_float": round(f.recall, 4),
                "rec_int8": round(q.recall, 4),
            }
        )
    rows.append(
        {
            "class": "MACRO",
            "support": sum(m.support for m in cmp.float_eval.per_class.values()),
            "f1_float": round(cmp.float_eval.macro_f1, 4),
            "f1_int8": round(cmp.quant_eval.macro_f1, 4),
            "df1": round(cmp.quant_eval.macro_f1 - cmp.float_eval.macro_f1, 4),
            "f2_float": round(cmp.float_eval.macro_f2, 4),
            "f2_int8": round(cmp.quant_eval.macro_f2, 4),
            "df2": round(cmp.quant_eval.macro_f2 - cmp.float_eval.macro_f2, 4),
            "prec_float": round(cmp.float_eval.macro_precision, 4),
            "prec_int8": round(cmp.quant_eval.macro_precision, 4),
            "rec_float": round(cmp.float_eval.macro_recall, 4),
            "rec_int8": round(cmp.quant_eval.macro_recall, 4),
        }
    )
    return pd.DataFrame(rows)


def plot_quantization_drop(
    cmp: QuantComparison,
    metric: Literal["f1", "f2", "precision", "recall"] = "f1",
) -> "matplotlib.figure.Figure":
    """Per-class side-by-side bars (float vs INT8) for metric, with the drop
    annotated above each pair."""
    import matplotlib.pyplot as plt

    names = cmp.float_eval.label_names
    f_vals = np.array(
        [getattr(cmp.float_eval.per_class[n], metric) for n in names]
    )
    q_vals = np.array(
        [getattr(cmp.quant_eval.per_class[n], metric) for n in names]
    )

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names) + 2), 4.5))
    ax.bar(x - w / 2, f_vals, w, label="float", color="#3b6fa3")
    ax.bar(x + w / 2, q_vals, w, label="INT8", color="#d97a3f")
    for xi, fv, qv in zip(x, f_vals, q_vals):
        ax.text(
            xi, max(fv, qv) + 0.01, f"{qv - fv:+.3f}",
            ha="center", va="bottom", fontsize=8, color="#444",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, min(1.05, max(f_vals.max(), q_vals.max()) + 0.12))
    ax.set_ylabel(metric)
    ax.set_title(f"Float vs INT8 quantized: {metric} per class")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def opti_table(opti: Any) -> "pd.DataFrame":
    """Per-class plus MACRO row table from one OptiResult (fixed
    backend/split/objective)."""
    import pandas as pd

    rows = [
        {
            "class": name, "support": m.support, "threshold": m.threshold,
            "precision": m.precision, "recall": m.recall,
            "f1": m.f1, "f2": m.f2, "accuracy": m.accuracy, "auc": m.auc,
        }
        for name, m in opti.per_class.items()
    ]
    mc = opti.macro
    rows.append({
        "class": "MACRO", "support": sum(r["support"] for r in rows),
        "threshold": None,
        "precision": mc.precision, "recall": mc.recall,
        "f1": mc.f1, "f2": mc.f2, "accuracy": mc.accuracy, "auc": mc.auc,
    })
    return pd.DataFrame(rows).set_index("class")


_OPTI_NUM_COLS = ("threshold", "precision", "recall", "f1", "f2", "accuracy", "auc")


def display_opti_tables(record: Any) -> None:
    """Render every (backend, split, objective) per-class table from
    record.opti_results."""
    from IPython.display import display

    for backend in ("keras", "tflite"):
        for split in ("train", "test"):
            for obj in ("f1", "f2"):
                opti = record.opti_results.get(backend, {}).get(split, {}).get(obj)
                if opti is None:
                    continue
                display(
                    opti_table(opti).style
                    .format({c: "{:.3f}" for c in _OPTI_NUM_COLS}, na_rep="-")
                    .format({"support": "{:,}"})
                    .set_caption(f"{backend} | {split} | thr-opt={obj.upper()}")
                )


def plot_prediction_rates(
    result: EvalResult,
) -> "matplotlib.figure.Figure":
    """Per-(true-class, predicted-head) firing rates at each class's tuned
    threshold. Rows are true class (K targets plus non_target), columns the K
    sigmoid heads; the KxK diagonal is recall. Rows can sum past 1 since heads
    are independent."""
    import matplotlib.pyplot as plt

    label_names = list(result.label_names)
    k = len(label_names)
    thr = np.array([result.per_class[n].threshold for n in label_names], dtype=np.float64)
    y_true_bin = (result.y_true >= 0.5).astype(np.int32)
    y_pred_bin = (result.y_score >= thr[None, :]).astype(np.int32)

    rates = np.zeros((k + 1, k), dtype=np.float64)
    for i in range(k):
        mask = y_true_bin[:, i] == 1
        if mask.any():
            rates[i] = y_pred_bin[mask].mean(axis=0)
    nt_mask = y_true_bin.sum(axis=1) == 0
    if nt_mask.any():
        rates[k] = y_pred_bin[nt_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(
        [f"{n}\n(thr={t:.2f})" for n, t in zip(label_names, thr)],
        rotation=30, ha="right",
    )
    ax.set_yticks(range(k + 1))
    ax.set_yticklabels(label_names + ["non_target"])
    ax.set_xlabel("Predicted class fires (>= per-class threshold)")
    ax.set_ylabel("True class")
    ax.set_title(f"Prediction rates ({result.backend}, per-species {result.threshold_mode})")
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            ax.text(
                j, i, f"{rates[i, j]:.2f}", ha="center", va="center",
                color="white" if rates[i, j] > 0.5 else "black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    confusion: dict[str, dict[str, list[list[int]]]],
    label_names: list[str],
) -> "matplotlib.figure.Figure":
    """Top-1 confusion grid over (backend, split). Matrices are (K+1)x(K+1);
    the trailing row/col is non_target."""
    import matplotlib.pyplot as plt

    cm_labels = list(label_names) + ["non_target"]
    backends = [b for b in ("keras", "tflite") if b in confusion]
    splits = [
        s for s in ("train", "test")
        if any(s in confusion.get(b, {}) for b in backends)
    ]
    fig, axes = plt.subplots(
        len(backends), len(splits),
        figsize=(4.5 * len(splits), 4 * len(backends)),
        squeeze=False,
    )
    for r, backend in enumerate(backends):
        for c, split in enumerate(splits):
            ax = axes[r][c]
            cm = confusion.get(backend, {}).get(split)
            if cm is None:
                ax.set_visible(False)
                continue
            cm = np.array(cm, dtype=int)
            row_sum = cm.sum(axis=1, keepdims=True)
            norm = np.divide(
                cm, row_sum,
                out=np.zeros_like(cm, dtype=float), where=row_sum > 0,
            )
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(cm_labels)))
            ax.set_xticklabels(cm_labels, rotation=30, ha="right")
            ax.set_yticks(range(len(cm_labels)))
            ax.set_yticklabels(cm_labels)
            ax.set_xlabel("predicted (top-1)")
            ax.set_ylabel("true")
            ax.set_title(f"{backend} | {split}")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black",
                        fontsize=9,
                    )
    fig.tight_layout()
    return fig


def print_comparison(cmp: QuantComparison) -> None:
    f, q = cmp.float_eval, cmp.quant_eval
    print(f"{'metric':<20s} {'float':>10s} {'INT8':>10s} {'delta':>10s}")
    print("-" * 52)

    def row(label: str, fv: float | None, qv: float | None) -> None:
        if fv is None or qv is None:
            return
        print(f"{label:<20s} {fv:>10.4f} {qv:>10.4f} {qv - fv:>+10.4f}")

    row("top-1 accuracy", f.top1_accuracy, q.top1_accuracy)
    row("subset accuracy", f.subset_accuracy, q.subset_accuracy)
    row("macro precision", f.macro_precision, q.macro_precision)
    row("macro recall", f.macro_recall, q.macro_recall)
    row("macro F1", f.macro_f1, q.macro_f1)
    row("macro F2", f.macro_f2, q.macro_f2)
    row("macro AUC", f.macro_auc, q.macro_auc)
    row("avg inference ms", f.avg_inference_ms, q.avg_inference_ms)
    if q.tflite_stats is not None:
        print("-" * 52)
        ts = q.tflite_stats
        print(f"{'flash (weights) KB':<20s} {'':>10s} {ts.model_size_kb:>10.1f}")
        if ts.arena_size_kb is not None:
            print(f"{'arena KB':<20s} {'':>10s} {ts.arena_size_kb:>10.1f}")
        print(f"{'MFLOPs (est.)':<20s} {'':>10s} {ts.flops_mflops:>10.3f}")
