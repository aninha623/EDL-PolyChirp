from __future__ import annotations

import gc
import os
from pathlib import Path
from time import perf_counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError
from tqdm.auto import tqdm
from datetime import datetime, timezone

from building.polychirp_training.training import (
    MacroBlock,
    RoundedFloat,
    TrainSidecar,
    build_dataset_from_catalog,
    build_datasets,
    featurize,
)

if TYPE_CHECKING:
    from building.polychirp_training import RunPaths
    from building.models.bake import TFLiteStats
    from building.models.model_eval import EvalResult

SWEEP_THRESHOLDS = np.linspace(0.0, 1.0, 101)

Backend = Literal["keras", "tflite"]
Split = Literal["train", "test"]      # `train` = train+val concatenated (the data the model "saw").
Objective = Literal["f1", "f2"]       # which Fbeta the per-class thresholds optimize.
BACKENDS: tuple[Backend, ...] = ("keras", "tflite")
SPLITS: tuple[Split, ...] = ("train", "test")
OBJECTIVES: tuple[Objective, ...] = ("f1", "f2")

# (y_true, y_score, avg_inference_ms) per backend per split.
Preds = dict[Backend, dict[Split, tuple[np.ndarray, np.ndarray, float]]]


class ClassMetric(BaseModel):
    """Per-class metrics at a chosen threshold. The threshold itself is stored
    here so the same OptiResult can be inspected stand-alone."""

    support: int
    threshold: RoundedFloat
    precision: RoundedFloat
    recall: RoundedFloat
    f1: RoundedFloat
    f2: RoundedFloat
    accuracy: RoundedFloat
    auc: RoundedFloat | None = None

    @classmethod
    def from_metric(cls, m: Any) -> "ClassMetric":
        return cls(
            support=int(m.support),
            threshold=float(m.threshold),
            precision=float(m.precision),
            recall=float(m.recall),
            f1=float(m.f1),
            f2=float(m.f2),
            accuracy=float(m.accuracy),
            auc=None if m.auc is None else float(m.auc),
        )


class OptiResult(BaseModel):
    """One evaluation block: per-class + macro metrics for a fixed
    (backend, split, optimized_on). The three discriminators live in the
    RunRecord.opti_results dict keys, not on this model."""

    per_class: dict[str, ClassMetric]
    macro: MacroBlock
    top1_accuracy: RoundedFloat
    subset_accuracy: RoundedFloat
    avg_inference_ms: RoundedFloat | None = None
    loss: RoundedFloat | None = None

    @classmethod
    def from_eval(
        cls, e: "EvalResult", *, loss: float | None = None
    ) -> "OptiResult":
        macro = MacroBlock(
            precision=float(e.macro_precision),
            recall=float(e.macro_recall),
            f1=float(e.macro_f1),
            f2=float(e.macro_f2),
            accuracy=float(e.macro_accuracy),
            auc=e.macro_auc,
        )
        return cls(
            per_class={n: ClassMetric.from_metric(m) for n, m in e.per_class.items()},
            macro=macro,
            top1_accuracy=float(e.top1_accuracy),
            subset_accuracy=float(e.subset_accuracy),
            avg_inference_ms=float(e.avg_inference_ms),
            loss=loss,
        )


# Nested-dict containers carried inside RunRecord. Pydantic serializes them as
# nested JSON objects keyed by the discriminator names.
OptiByObjective = dict[Objective, OptiResult]
OptiBySplit = dict[Split, OptiByObjective]
OptiByBackend = dict[Backend, OptiBySplit]

ConfusionBySplit = dict[Split, list[list[int]]]
ConfusionByBackend = dict[Backend, ConfusionBySplit]


class TFLiteStatsBlock(BaseModel):
    path: str
    model_size_kb: RoundedFloat
    arena_size_kb: RoundedFloat | None = None
    flops_mflops: RoundedFloat
    input_dtype: str
    output_dtype: str
    input_shape: list[int]

    @classmethod
    def from_stats(cls, stats: "TFLiteStats | None") -> "TFLiteStatsBlock | None":
        if stats is None:
            return None
        return cls(
            path=str(stats.path),
            model_size_kb=float(stats.model_size_kb),
            arena_size_kb=(
                None if stats.arena_size_kb is None else float(stats.arena_size_kb)
            ),
            flops_mflops=float(stats.flops_mflops),
            input_dtype=str(stats.input_dtype),
            output_dtype=str(stats.output_dtype),
            input_shape=[int(s) for s in stats.input_shape],
        )


class ThresholdCurve(BaseModel):
    threshold: list[RoundedFloat]
    precision: list[RoundedFloat]
    recall: list[RoundedFloat]
    f1: list[RoundedFloat]
    f2: list[RoundedFloat]
    accuracy: list[RoundedFloat]


class CurvesBlock(BaseModel):
    thresholds: list[RoundedFloat]
    per_class: dict[str, ThresholdCurve]

    @classmethod
    def from_eval(cls, e: "EvalResult") -> "CurvesBlock":
        from building.models.model_eval import threshold_curve

        per_class: dict[str, ThresholdCurve] = {}
        for c, name in enumerate(e.label_names):
            yt = e.y_true[:, c]
            if (yt >= 0.5).sum() == 0:
                continue
            curve = threshold_curve(yt, e.y_score[:, c], SWEEP_THRESHOLDS)
            per_class[name] = ThresholdCurve(
                threshold=curve["threshold"].tolist(),
                precision=curve["precision"].tolist(),
                recall=curve["recall"].tolist(),
                f1=curve["f1"].tolist(),
                f2=curve["f2"].tolist(),
                accuracy=curve["accuracy"].tolist(),
            )
        return cls(
            thresholds=SWEEP_THRESHOLDS.tolist(),
            per_class=per_class,
        )


class ROCCurve(BaseModel):
    fpr: list[RoundedFloat]
    tpr: list[RoundedFloat]
    auc: RoundedFloat


def roc_block(e: "EvalResult") -> dict[str, ROCCurve]:
    from building.models.model_eval import roc_arrays

    out: dict[str, ROCCurve] = {}
    for c, name in enumerate(e.label_names):
        r = roc_arrays(e.y_true[:, c], e.y_score[:, c])
        if r["auc"] is None:
            continue
        out[name] = ROCCurve(
            fpr=[float(v) for v in r["fpr"]],
            tpr=[float(v) for v in r["tpr"]],
            auc=float(r["auc"]),
        )
    return out


def prediction_rates(e: "EvalResult") -> list[list[float]]:
    # Diagonal = recall; off-diagonal cells can sum past 1 (predictions are
    # independent multi-label sigmoids).
    n = len(e.label_names)
    y_true_bin = (e.y_true >= 0.5).astype(np.int32)
    thresholds = np.array(
        [e.per_class[name].threshold for name in e.label_names], dtype=float
    )
    y_pred_bin = (e.y_score >= thresholds[None, :]).astype(np.int32)
    rates = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mask = y_true_bin[:, i] == 1
        if mask.any():
            rates[i] = y_pred_bin[mask].mean(axis=0)
    return rates.tolist()


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    opti_results: OptiByBackend
    # Top-1 confusion per (backend, split). Identical across f1/f2, so hoisted.
    confusion: ConfusionByBackend

    tflite_stats: TFLiteStatsBlock | None = None
    # Threshold-sweep curves and ROC for the quantized backend on the test split;
    # supports drill-down plots. Not duplicated for other combos to keep file size
    # down; the raw scores are in the .npz for recomputation.
    curves_quantized: CurvesBlock
    roc_quantized: dict[str, ROCCurve] = Field(default_factory=dict)
    prediction_rates_quantized: list[list[RoundedFloat]] | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Run metadata.
    collection: str | None = None
    model_path: str | None = None
    tflite_path: str | None = None
    # Typed training params for the run.
    train: TrainSidecar | None = None
    # Free-form evaluator knobs only (e.g. n_representative); training params live in `train`.
    hyperparams: dict[str, Any] = Field(default_factory=dict)

    # Transient, never serialized. Set by from_evaluations so save() can flush
    # the raw arrays without the caller having to thread them through again.
    _evals: "dict[Backend, dict[Split, EvalResult]] | None" = PrivateAttr(default=None)
    _extra_arrays: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_evaluations(
        cls,
        evals: "dict[Backend, dict[Split, dict[Objective, EvalResult]]]",
        *,
        losses: dict[Backend, dict[Split, float | None]] | None = None,
        tflite_stats: "TFLiteStats | None" = None,
        extra_arrays: dict[str, np.ndarray] | None = None,
        **meta: Any,
    ) -> "RunRecord":
        opti: OptiByBackend = {}
        confusion: ConfusionByBackend = {}
        # One EvalResult per (backend, split); same arrays, threshold vector
        # differs by objective. Used to rebuild test-only curves and raw arrays.
        canonical: dict[Backend, dict[Split, "EvalResult"]] = {}
        for backend, by_split in evals.items():
            opti[backend] = {}
            confusion[backend] = {}
            canonical[backend] = {}
            for split, by_obj in by_split.items():
                opti[backend][split] = {}
                first = next(iter(by_obj.values()))
                confusion[backend][split] = first.confusion.tolist()
                canonical[backend][split] = first
                for obj, ev in by_obj.items():
                    loss = (losses or {}).get(backend, {}).get(split)
                    opti[backend][split][obj] = OptiResult.from_eval(ev, loss=loss)

        # Test-quant carries the curves/ROC/prediction-rates we still serialize.
        quant_test = canonical.get("tflite", {}).get("test")
        if quant_test is None:
            raise ValueError(
                "from_evaluations needs a tflite/test EvalResult to derive "
                "curves_quantized / roc_quantized / prediction_rates."
            )

        payload: dict[str, Any] = {
            "opti_results": opti,
            "confusion": confusion,
            "tflite_stats": TFLiteStatsBlock.from_stats(tflite_stats),
            "curves_quantized": CurvesBlock.from_eval(quant_test),
            "roc_quantized": roc_block(quant_test),
            "prediction_rates_quantized": prediction_rates(quant_test),
        }
        payload.update(meta)
        rec = cls.model_validate(payload)
        rec._evals = canonical
        if extra_arrays:
            rec._extra_arrays = dict(extra_arrays)
        return rec

    def save(
        self,
        results_file: Path | str,
        *,
        with_arrays: bool = True,
        npz_file: Path | str | None = None,
    ) -> Path:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(self.model_dump_json(indent=2, by_alias=True))
        if with_arrays:
            if self._evals is None:
                raise ValueError(
                    "Cannot write arrays, record was not built from evaluations."
                )
            npz_path = (
                Path(npz_file)
                if npz_file is not None
                else results_path.with_suffix(".npz")
            )
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = build_arrays_npz(evals=self._evals)
            arrays.update(self._extra_arrays)
            np.savez_compressed(npz_path, **arrays)
        return results_path


def build_arrays_npz(
    *,
    evals: "dict[Backend, dict[Split, EvalResult]]",
) -> dict[str, np.ndarray]:
    """Persist raw `y_true` / `y_score` for every (backend, split) pair, plus
    the test-quant per-class ROC/sweep arrays the drill-down plots consume."""
    from building.models.model_eval import roc_arrays, threshold_curve

    quant_test = evals["tflite"]["test"]
    out: dict[str, np.ndarray] = {
        "label_names": np.array(quant_test.label_names),
        "thresholds": SWEEP_THRESHOLDS.copy(),
    }
    for backend, by_split in evals.items():
        bk = "float" if backend == "keras" else "quant"
        for split, ev in by_split.items():
            out[f"y_true_{split}"] = ev.y_true.astype(np.float32)
            out[f"y_score_{bk}_{split}"] = ev.y_score.astype(np.float32)
            out[f"avg_ms_{bk}_{split}"] = np.float32(ev.avg_inference_ms)

    float_test = evals.get("keras", {}).get("test")
    for c, name in enumerate(quant_test.label_names):
        yt = quant_test.y_true[:, c]
        ysq = quant_test.y_score[:, c]
        if (yt >= 0.5).sum() == 0:
            continue
        r = roc_arrays(yt, ysq)
        if r["auc"] is not None:
            out[f"roc_fpr__{name}"] = r["fpr"].astype(np.float32)
            out[f"roc_tpr__{name}"] = r["tpr"].astype(np.float32)
            out[f"roc_thresholds__{name}"] = r["thresholds"].astype(np.float32)
        pairs: list[tuple[str, np.ndarray]] = [("quant", ysq)]
        if float_test is not None:
            pairs.append(("float", float_test.y_score[:, c]))
        for label, ys in pairs:
            curve = threshold_curve(yt, ys, SWEEP_THRESHOLDS)
            for metric, arr in curve.items():
                if metric == "threshold":
                    continue
                out[f"sweep_{label}_{metric}__{name}"] = arr.astype(np.float32)
    return out




class LoadedRun(BaseModel):
    """A `RunRecord` paired with the raw arrays and rehydrated EvalResults for
    every (backend, split, objective). Drill-down notebooks index into
    `evals[backend][split][objective]` (or use the `eval(...)` shortcut)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    record: RunRecord
    evals: dict[Backend, dict[Split, dict[Objective, Any]]]  # EvalResult
    label_names: list[str]
    npz: Any  # numpy.lib.npyio.NpzFile

    def eval(
        self,
        backend: Backend = "tflite",
        split: Split = "test",
        objective: Objective = "f2",
    ) -> Any:
        return self.evals[backend][split][objective]

    def cmp(self, split: Split = "test", objective: Objective = "f2") -> Any:
        from building.models.model_eval import QuantComparison
        return QuantComparison(
            float_eval=self.evals["keras"][split][objective],
            quant_eval=self.evals["tflite"][split][objective],
        )

    @property
    def losses(self) -> dict[Backend, dict[Split, float]]:
        """Same surface as BakedRun.losses. Per-sample loss is threshold-free, so
        any OptiResult in (backend, split) carries the same value; read it off the
        first objective."""
        out: dict[Backend, dict[Split, float]] = {}
        for backend, by_split in self.record.opti_results.items():
            out[backend] = {}
            for split, by_obj in by_split.items():
                first = next(iter(by_obj.values()))
                if first.loss is not None:
                    out[backend][split] = float(first.loss)
        return out

    @property
    def meta(self) -> dict[str, Any]:
        return self.record.model_dump(mode="json", by_alias=True)


class BakedRun(BaseModel):
    """Return value of bake_eval_save: everything needed to render results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    record: RunRecord
    evals: dict[Backend, dict[Split, dict[Objective, Any]]]  # EvalResult
    tflite_stats: Any  # building.models.bake.TFLiteStats
    losses: dict[Backend, dict[Split, float]]
    results_file: Path

    def eval(
        self,
        backend: Backend = "tflite",
        split: Split = "test",
        objective: Objective = "f2",
    ) -> Any:
        return self.evals[backend][split][objective]

    def cmp(self, split: Split = "test", objective: Objective = "f2") -> Any:
        from building.models.model_eval import QuantComparison
        return QuantComparison(
            float_eval=self.evals["keras"][split][objective],
            quant_eval=self.evals["tflite"][split][objective],
        )

    @property
    def label_names(self) -> list[str]:
        return list(self.evals["tflite"]["test"]["f2"].label_names)

    @property
    def meta(self) -> dict[str, Any]:
        return self.record.model_dump(mode="json", by_alias=True)


def bake_for_run(
    model: Any, *, val_x: np.ndarray, tflite_path: Path, n_representative: int
) -> "TFLiteStats":
    """PTQ-bake `model` to INT8, or reuse an existing flatbuffer; return its
    footprint stats. Calibrates on the shared featurized val features."""
    import tensorflow as tf
    from building.models.bake import analyze_tflite, bake_model

    if tflite_path.exists():
        print(f"  reusing baked tflite {tflite_path.name}")
        return analyze_tflite(tflite_path)
    rep_ds = tf.data.Dataset.from_tensor_slices(val_x).batch(1)
    return bake_model(
        model, rep_ds, tflite_path, n_representative=n_representative, verbose=False
    )


def load_cached_preds(npz_path: Path, label_names: list[str]) -> Preds | None:
    """Rebuild the predictions dict from a cached npz when it matches the current
    label set, else None, so a catch-up rerun recomputes only the JSON metrics."""
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=False)
    required = {
        "label_names",
        "y_true_train", "y_true_test",
        "y_score_float_train", "y_score_float_test",
        "y_score_quant_train", "y_score_quant_test",
    }
    if not (
        required.issubset(npz.files)
        and [str(n) for n in npz["label_names"]] == label_names
    ):
        return None
    ms = lambda k: float(npz[k]) if k in npz.files else 0.0
    print(f"  reusing cached predictions {npz_path.name}")
    return {
        "keras": {
            "train": (npz["y_true_train"], npz["y_score_float_train"], ms("avg_ms_float_train")),
            "test": (npz["y_true_test"], npz["y_score_float_test"], ms("avg_ms_float_test")),
        },
        "tflite": {
            "train": (npz["y_true_train"], npz["y_score_quant_train"], ms("avg_ms_quant_train")),
            "test": (npz["y_true_test"], npz["y_score_quant_test"], ms("avg_ms_quant_test")),
        },
    }


def predict_for_run(
    model: Any,
    tflite_path: Path,
    *,
    tune_x: np.ndarray,
    tune_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    q_tune: tuple[np.ndarray, float] | None = None,
    q_test: tuple[np.ndarray, float] | None = None,
) -> Preds:
    """Float predictions (serial, batched) + INT8 predictions over the shared
    featurized arrays. Pass `q_tune`/`q_test` (from a parallel
    `predict_tflite_array` pass) to skip the in-line INT8 sweep."""
    from building.models import model_eval as M

    ys_tune_f, ms_tune_f = M.predict_keras_array(model, tune_x)
    ys_test_f, ms_test_f = M.predict_keras_array(model, test_x)
    ys_tune_q, ms_tune_q = q_tune or M.predict_tflite_array(tflite_path, tune_x)
    ys_test_q, ms_test_q = q_test or M.predict_tflite_array(tflite_path, test_x)
    return {
        "keras": {
            "train": (tune_y, ys_tune_f, ms_tune_f),
            "test": (test_y, ys_test_f, ms_test_f),
        },
        "tflite": {
            "train": (tune_y, ys_tune_q, ms_tune_q),
            "test": (test_y, ys_test_q, ms_test_q),
        },
    }


def eval_and_save(
    preds: Preds,
    *,
    stats: "TFLiteStats",
    label_names: Iterable[str],
    paths: "RunPaths",
    train: TrainSidecar,
    n_representative: int,
    extra_meta: dict[str, Any] | None = None,
) -> "BakedRun":
    """Tune per-class thresholds on the train+val split, evaluate float + INT8
    under F1- and F2-optimal thresholds, write JSON + npz. Per-class thresholds
    are picked independently per backend (each deploys at its own optimum)."""
    import tensorflow as tf
    from building.models import model_eval as M

    labels = list(label_names)
    bce = tf.keras.losses.BinaryCrossentropy()
    losses: dict[Backend, dict[Split, float]] = {
        backend: {
            split: float(bce(yt, ys).numpy())
            for split, (yt, ys, _) in by_split.items()
        }
        for backend, by_split in preds.items()
    }

    evals: dict[Backend, dict[Split, dict[Objective, Any]]] = {}
    for backend in BACKENDS:
        yt_tune, ys_tune, _ = preds[backend]["train"]
        thresholds = {
            "f1": M.pick_best_thresholds(yt_tune, ys_tune, labels, "best_f1"),
            "f2": M.pick_best_thresholds(yt_tune, ys_tune, labels, "best_f2"),
        }
        evals[backend] = {}
        for split in SPLITS:
            yt, ys, avg_ms = preds[backend][split]
            evals[backend][split] = {}
            for obj in OBJECTIVES:
                evals[backend][split][obj] = M.eval_with_thresholds(
                    backend=backend,
                    label_names=labels,
                    y_true=yt,
                    y_score=ys,
                    thresholds=thresholds[obj],
                    avg_inference_ms=avg_ms,
                    threshold_mode=f"best_{obj}",
                    tflite_stats=stats if backend == "tflite" else None,
                )

    record = RunRecord.from_evaluations(
        evals,
        losses=losses,
        tflite_stats=stats,
        collection=train.collection,
        model_path=str(paths.model_path),
        tflite_path=str(paths.tflite_path),
        train=train,
        hyperparams={"n_representative": n_representative},
        **(extra_meta or {}),
    )
    results_file = Path(paths.results_path)
    record.save(results_file)
    return BakedRun(
        record=record,
        evals=evals,
        tflite_stats=stats,
        losses=losses,
        results_file=results_file,
    )


def dataset_to_arrays(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate an already-featurized (features, label[, sw]) dataset into
    (X, Y) numpy arrays without re-featurizing."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch in ds:
        xs.append(batch[0].numpy())
        ys.append(batch[1].numpy())
    return np.concatenate(xs), np.concatenate(ys)


def bake_eval_save(
    model: Any,
    *,
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    label_names: Iterable[str],
    paths: "RunPaths",
    train: TrainSidecar,
    n_representative: int = 2000,
    extra_meta: dict[str, Any] | None = None,
) -> "BakedRun":
    """Bake INT8, evaluate float and INT8 on train+val and test under both F1 and
    F2-optimal thresholds, then write JSON + npz, from already-featurized datasets.

    Idempotent: skips the PTQ rebake when paths.tflite_path exists and reuses
    cached predictions from paths.npz_path, so a catch-up rerun after a schema
    bump only recomputes the JSON metrics.

    train_ds must be the clean (no augment) train split. val_ds is the PTQ
    calibration source and is concatenated with train_ds to form the "train" eval
    split where thresholds are tuned. test_ds is held out."""
    labels = list(label_names)
    tflite_path = Path(paths.tflite_path)

    train_x, train_y = dataset_to_arrays(train_ds)
    val_x, val_y = dataset_to_arrays(val_ds)
    test_x, test_y = dataset_to_arrays(test_ds)

    stats = bake_for_run(
        model, val_x=val_x, tflite_path=tflite_path, n_representative=n_representative
    )
    preds = load_cached_preds(Path(paths.npz_path), labels) or predict_for_run(
        model, tflite_path,
        tune_x=np.concatenate([train_x, val_x]),
        tune_y=np.concatenate([train_y, val_y]),
        test_x=test_x, test_y=test_y,
    )
    return eval_and_save(
        preds, stats=stats, label_names=labels, paths=paths, train=train,
        n_representative=n_representative, extra_meta=extra_meta,
    )


def collect_representative(ds: Any, repr_: Any, n_max: int) -> np.ndarray:
    """Stream up to n_max featurized samples out of `ds` for PTQ calibration,
    bounded so a large val split doesn't materialize end-to-end."""
    chunks: list[np.ndarray] = []
    n = 0
    for batch in ds:
        feats = featurize(batch[0], repr_).numpy()
        take = min(len(feats), n_max - n)
        chunks.append(feats[:take])
        n += take
        if n >= n_max:
            break
    return np.concatenate(chunks)


def make_tflite_runner(tflite_path: Path | str, num_threads: int = 1):
    """Build a per-interpreter callable that quantizes, invokes, and dequantizes
    one sample at a time. Returned as (interpreter, run(sample) -> score). Kept as
    a closure so stream_predict reuses one interpreter across batches.

    num_threads is sized by the caller: stream_predict keeps every model in a repr
    group alive at once, so n interpreters each with their own threadpool would
    oversubscribe the cores and spin-wait against each other (~24x slower per
    invoke). Splitting (cpu-4) across the group keeps the total at the core budget
    while a lone model still gets the full pool."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_dtype, out_dtype = inp["dtype"], out["dtype"]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    in_idx, out_idx = inp["index"], out["index"]
    if in_dtype == np.float32:
        quantize = lambda s: s.astype(np.float32)[None, ...]
    elif in_dtype in (np.int8, np.uint8):
        qmin, qmax = (-128, 127) if in_dtype == np.int8 else (0, 255)
        quantize = lambda s: np.clip(np.round(s / in_scale) + in_zp, qmin, qmax).astype(in_dtype)[None, ...]
    elif in_dtype == np.int16:
        quantize = lambda s: np.clip(np.round(s / in_scale) + in_zp, -32768, 32767).astype(np.int16)[None, ...]
    else:
        raise ValueError(f"{tflite_path} has unsupported input dtype {in_dtype}")
    if out_dtype == np.float32:
        dequantize = lambda raw: raw.astype(np.float32)
    else:
        dequantize = lambda raw: (raw.astype(np.float32) - out_zp) * out_scale

    def run(sample: np.ndarray) -> np.ndarray:
        interp.set_tensor(in_idx, quantize(sample))
        interp.invoke()
        return dequantize(interp.get_tensor(out_idx)).reshape(-1)

    return interp, run


def stream_predict(
    items: list[tuple[Any, "RunPaths", TrainSidecar]],
    pending: list[int],
    datasets: list[Any],
    repr_: Any,
    *,
    desc: str,
    verbose: bool = True,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """Walk every raw-audio batch in `datasets` once: featurize, run every model's
    float Keras predict (batched) and INT8 invoke (per-sample), and accumulate
    only the small (N, n_classes) score arrays. Peak memory is one batch of
    features plus the per-model score lists, never the full split.

    Returns {i: (y_true, score_float, score_quant, avg_ms_float, avg_ms_quant)}.
    """
    nthr = max(1, (os.cpu_count() - 4) // len(pending)) if pending else 1
    runners = {i: make_tflite_runner(items[i][1].tflite_path, nthr) for i in pending}
    y_true_chunks: list[np.ndarray] = []
    f_chunks: dict[int, list[np.ndarray]] = {i: [] for i in pending}
    q_chunks: dict[int, list[np.ndarray]] = {i: [] for i in pending}
    f_time: dict[int, float] = {i: 0.0 for i in pending}
    q_time: dict[int, float] = {i: 0.0 for i in pending}
    n_samples = 0
    bar = tqdm if verbose else (lambda it, **_k: it)
    try:
        for ds in datasets:
            for batch in bar(ds, desc=desc, unit="batch", leave=False):
                feats = featurize(batch[0], repr_).numpy()
                y_true_chunks.append(batch[1].numpy())
                n_samples += len(feats)
                for i in pending:
                    model = items[i][0]
                    t0 = perf_counter()
                    f_chunks[i].append(model(feats, training=False).numpy())
                    f_time[i] += perf_counter() - t0
                for s in feats:
                    for i in pending:
                        _, run = runners[i]
                        t0 = perf_counter()
                        q_chunks[i].append(run(s))
                        q_time[i] += perf_counter() - t0
                del feats
    finally:
        runners.clear()
    y_true = np.concatenate(y_true_chunks)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    for i in pending:
        avg_ms_f = (f_time[i] / n_samples) * 1000.0 if n_samples else 0.0
        avg_ms_q = (q_time[i] / n_samples) * 1000.0 if n_samples else 0.0
        out[i] = (y_true, np.concatenate(f_chunks[i]), np.asarray(q_chunks[i]), avg_ms_f, avg_ms_q)
    return out


def tune_cell(
    items: list[tuple[Any, "RunPaths", TrainSidecar]],
    *,
    train_eval_ds: Any,
    val_ds: Any,
    test_ds: Any,
    label_names: Iterable[str],
    n_representative: int = 2000,
    extra_meta: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list["BakedRun"]:
    """Tune a whole cell of co-trained models together, streaming the shared raw
    eval datasets. Each split is walked once per input_repr; for every batch we
    featurize once and dispatch to every model in the group (Keras float and
    TFLite INT8) before moving on, so features are never materialized into
    per-split arrays and no work is duplicated across models.

    items is (model, paths, sidecar) per already-snapshotted model. The same
    extra_meta is attached to every run. Idempotent per model (skip-if-tflite,
    reuse-npz)."""
    labels = list(label_names)
    if not items:
        return []
    bar = tqdm if verbose else (lambda it, **_k: it)

    by_repr: dict[Any, list[int]] = {}
    for i, (_, _, sc) in enumerate(items):
        by_repr.setdefault(sc.model.input_frontend, []).append(i)

    baked: list[BakedRun | None] = [None] * len(items)
    for repr_, idxs in by_repr.items():
        tag = f"[tune] repr={repr_} ({len(idxs)} model{'s' if len(idxs) > 1 else ''})"
        names = ", ".join(
            f"{items[i][2].model.name}x{items[i][2].model.multi:g}" for i in idxs
        )
        if verbose:
            print(f"{tag}: {names}")

        t0 = perf_counter()
        rep_x = collect_representative(val_ds, repr_, n_representative)
        stats = {
            i: bake_for_run(
                items[i][0], val_x=rep_x, tflite_path=Path(items[i][1].tflite_path),
                n_representative=n_representative,
            )
            for i in bar(idxs, desc=f"{tag} bake INT8", unit="model", leave=False)
        }
        del rep_x
        gc.collect()
        if verbose:
            print(f"{tag} baked {len(idxs)} model(s) in {perf_counter() - t0:.1f}s")

        cached = {i: load_cached_preds(Path(items[i][1].npz_path), labels) for i in idxs}
        pending = [i for i in idxs if cached[i] is None]
        scored: dict[int, Preds] = {}
        if pending:
            t0 = perf_counter()
            tune = stream_predict(items, pending, [train_eval_ds, val_ds], repr_, desc=f"{tag} tune", verbose=verbose)
            test = stream_predict(items, pending, [test_ds], repr_, desc=f"{tag} test", verbose=verbose)
            if verbose:
                print(f"{tag} predicted in {perf_counter() - t0:.1f}s")
            for i in pending:
                ty, tf_, tq, tmsf, tmsq = tune[i]
                ey, ef, eq, emsf, emsq = test[i]
                scored[i] = {
                    "keras": {"train": (ty, tf_, tmsf), "test": (ey, ef, emsf)},
                    "tflite": {"train": (ty, tq, tmsq), "test": (ey, eq, emsq)},
                }
            del tune, test

        t0 = perf_counter()
        for i in bar(idxs, desc=f"{tag} eval+save", unit="model", leave=False):
            _, paths, sc = items[i]
            preds = cached[i] if cached[i] is not None else scored[i]
            baked[i] = eval_and_save(
                preds, stats=stats[i], label_names=labels, paths=paths, train=sc,
                n_representative=n_representative, extra_meta=extra_meta,
            )
        if verbose:
            print(f"{tag} eval+save done in {perf_counter() - t0:.1f}s")

        del cached, scored, stats
        gc.collect()
    return baked


def load_run(
    json_path: Path | str,
    *,
    npz_path: Path | str | None = None,
) -> LoadedRun:
    from building.models.bake import TFLiteStats
    from building.models.model_eval import eval_with_thresholds

    jp = Path(json_path)
    record = RunRecord.model_validate_json(jp.read_text())

    np_path = Path(npz_path) if npz_path is not None else jp.with_suffix(".npz")
    if not np_path.exists():
        raise FileNotFoundError(f"NPZ sidecar not found: {np_path}")
    npz = np.load(np_path, allow_pickle=False)

    label_names = [str(n) for n in npz["label_names"]]

    tflite_stats: TFLiteStats | None = None
    if record.tflite_stats is not None:
        tflite_stats = TFLiteStats.model_validate(
            record.tflite_stats.model_dump(mode="python")
        )

    evals: dict[Backend, dict[Split, dict[Objective, Any]]] = {}
    for backend, by_split in record.opti_results.items():
        bk_key = "float" if backend == "keras" else "quant"
        evals[backend] = {}
        for split, by_obj in by_split.items():
            y_true = npz[f"y_true_{split}"]
            y_score = npz[f"y_score_{bk_key}_{split}"]
            evals[backend][split] = {}
            for obj, opti in by_obj.items():
                thresholds = np.array(
                    [
                        float(opti.per_class[n].threshold) if n in opti.per_class else 0.5
                        for n in label_names
                    ],
                    dtype=float,
                )
                evals[backend][split][obj] = eval_with_thresholds(
                    backend=backend,
                    label_names=label_names,
                    y_true=y_true,
                    y_score=y_score,
                    thresholds=thresholds,
                    threshold_mode=f"best_{obj}",
                    avg_inference_ms=float(opti.avg_inference_ms or 0.0),
                    tflite_stats=tflite_stats if backend == "tflite" else None,
                )
    return LoadedRun(
        record=record,
        evals=evals,
        label_names=label_names,
        npz=npz,
    )


# Cross-run gather and task co-training driver.

MACRO_FIELDS = ("precision", "recall", "f1", "f2", "accuracy", "auc")
PER_CLASS_METRICS = ("precision", "recall", "f1", "f2", "accuracy", "auc")


def run_label(path: Path) -> str:
    """<model>/size_x<mult>, prefixed with k<k>/sample_<i>/ for a scale cell."""
    parts = path.parent.parts
    for i, name in enumerate(parts):
        if name.startswith("size_x"):
            label = f"{parts[i - 1]}/{name}"
            if i >= 3 and parts[i - 2].startswith("sample_") and parts[i - 3].startswith("k"):
                label = f"{parts[i - 3]}/{parts[i - 2]}/{label}"
            return label
    return f"{path.parent.parent.name}/{path.parent.name}"


class RunRow(BaseModel):
    """Flat row for cross-run comparison. Dynamic per-species columns
    (f1__<species>, train_macro_f2, ...) ride on model_extra."""

    model_config = ConfigDict(extra="allow")

    file: str
    collection: str | None = None
    variant: Literal["standard"] = "standard"
    model: str
    build_model: str | None = None
    multiplier: float | None = None

    subset_accuracy: float | None = None
    top1_accuracy: float | None = None
    test_loss: float | None = None
    test_loss_float: float | None = None
    delta_loss: float | None = None
    epochs_trained: int | None = None
    timestamp: str | None = None

    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    macro_f2: float | None = None
    macro_auc: float | None = None

    flash_kb: float | None = None
    arena_kb: float | None = None
    mflops: float | None = None

    def put(self, key: str, value: object) -> None:
        if self.model_extra is not None:
            self.model_extra[key] = value


PAPER_NAMES = {
    "cnn1d": "CNN-Time",
    "transformer": "Transformer-Time",
    "mel_cnn": "CNN-Mel",
    "mel_ultralight": "Mel-PolyChirp",
    "mel_polychirp": "Mel-PolyChirp",
    "ds_cnn": "DS-CNN",
    "wrennet": "WrenNet",
    "sincnet_norm": "SincNet",
    "leaf_pcen": "LEAF",
}


def run_model_label(name: str, multiplier: float) -> str:
    """Paper display label for one model run, e.g. CNN-Mel or Mel-PolyChirp×2."""
    base = PAPER_NAMES.get(name, name)
    return base if multiplier == 1.0 else f"{base}×{multiplier:g}"


def runrecord_to_row(rec: RunRecord, path: Path) -> RunRow:
    """Flatten the canonical (tflite, test, f2) cell into the cross-run row.

    The other 7 OptiResults are still on the record; the drill-down notebook
    reads them directly. This row is for the cross-run gather table."""
    train = rec.train
    multiplier = train.model.multi if train else None
    base_model = train.model.name if train else None
    model_label = run_model_label(base_model or "?", multiplier or 1.0)

    quant = rec.opti_results.get("tflite", {}).get("test", {}).get("f2")
    flt = rec.opti_results.get("keras", {}).get("test", {}).get("f2")
    qm, fm = (quant.macro if quant else None), (flt.macro if flt else None)
    q_loss = quant.loss if quant else None
    f_loss = flt.loss if flt else None

    row = RunRow(
        file=run_label(path),
        collection=rec.collection,
        variant="standard",
        model=model_label,
        build_model=base_model,
        multiplier=multiplier,
        subset_accuracy=quant.subset_accuracy if quant else None,
        top1_accuracy=quant.top1_accuracy if quant else None,
        test_loss=q_loss,
        test_loss_float=f_loss,
        delta_loss=(q_loss - f_loss) if (q_loss is not None and f_loss is not None) else None,
        epochs_trained=train.epochs_trained if train else None,
        timestamp=rec.timestamp,
        macro_precision=qm.precision if qm else None,
        macro_recall=qm.recall if qm else None,
        macro_f1=qm.f1 if qm else None,
        macro_f2=qm.f2 if qm else None,
        macro_auc=qm.auc if qm else None,
        flash_kb=rec.tflite_stats.model_size_kb if rec.tflite_stats else None,
        arena_kb=rec.tflite_stats.arena_size_kb if rec.tflite_stats else None,
        mflops=rec.tflite_stats.flops_mflops if rec.tflite_stats else None,
    )
    for k in MACRO_FIELDS:
        row.put(f"macro_{k}", getattr(qm, k, None) if qm else None)
        row.put(f"float_macro_{k}", getattr(fm, k, None) if fm else None)

    quant_train = rec.opti_results.get("tflite", {}).get("train", {}).get("f2")
    qm_tr = quant_train.macro if quant_train else None
    for k in MACRO_FIELDS:
        row.put(f"train_macro_{k}", getattr(qm_tr, k, None) if qm_tr else None)

    if quant is not None:
        for name, m in quant.per_class.items():
            for metric in PER_CLASS_METRICS:
                row.put(f"{metric}__{name}", getattr(m, metric, None))
            row.put(f"support__{name}", m.support)
            row.put(f"threshold__{name}", m.threshold)
    # Grid coords carried by geographic-scale runs (None for task runs).
    extra = rec.model_extra or {}
    row.put("k", extra.get("k"))
    row.put("sample_idx", extra.get("sample_idx"))
    return row


SCHEMA_CANDIDATES = (
    (RunRecord, runrecord_to_row),
)


def load_run_row(path: Path) -> RunRow | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    for model_cls, to_row in SCHEMA_CANDIDATES:
        try:
            rec = model_cls.model_validate_json(text)
        except ValidationError:
            continue
        return to_row(rec, path)
    return None


def gather_results(results_dir: Path) -> pd.DataFrame:
    rows = [r.model_dump() for p in sorted(results_dir.rglob("results.json")) if (r := load_run_row(p))]
    df = pd.DataFrame(rows)
    sort_keys = [k for k in ("macro_f1", "macro_f2") if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, ascending=False, na_position="last").reset_index(drop=True)
    return df


def gather_macro_table(
    results_dir: Path,
    backend: Backend,
    objective: Objective,
    *,
    split: Split = "test",
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "f2", "auc"),
) -> pd.DataFrame:
    """One row per run: macro metrics for a fixed (backend, split, objective)
    plus flash size, indexed by model label and sorted by the objective. Carries
    build_model/multiplier so callers can filter to a curated model set."""
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("results.json")):
        try:
            rec = RunRecord.model_validate_json(path.read_text())
        except (OSError, ValidationError):
            continue
        opti = rec.opti_results.get(backend, {}).get(split, {}).get(objective)
        if opti is None:
            continue
        train = rec.train
        name = train.model.name if train else "?"
        multi = train.model.multi if train else 1.0
        row: dict[str, Any] = {
            "model": run_model_label(name, multi),
            "build_model": name,
            "multiplier": multi,
            "flash_kb": rec.tflite_stats.model_size_kb if rec.tflite_stats else None,
        }
        row.update({f"macro_{k}": getattr(opti.macro, k) for k in metrics})
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_key = f"macro_{objective}"
    if sort_key in df.columns:
        df = df.sort_values(sort_key, ascending=False)
    return df.set_index("model")


def per_class_metric_table(
    df: pd.DataFrame,
    metric: Literal["precision", "recall", "f1", "f2", "auc"] = "f1",
) -> pd.DataFrame:
    prefix = f"{metric}__"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    out = df.set_index(["variant", "model"])[cols].copy()
    out.columns = [c.removeprefix(prefix) for c in out.columns]
    return out


per_class_f1_table = per_class_metric_table


def build_eval_datasets(catalog: Any, sidecar: Any) -> tuple[Any, Any, Any, list[str]]:
    """Build clean (no augment) train/val/test datasets matching a TrainSidecar.

    Used by both the end-of-training PTQ cell and the catch-up loop so they
    walk the catalog through the same code path."""
    import numpy as np

    non_target_idx = catalog.class_names.index("non_target")
    target_idxs = [
        i for i, n in enumerate(catalog.class_names) if n != "non_target"
    ]
    rng = np.random.default_rng(sidecar.seed)
    train_eval_ds, val_ds, test_ds, final_labels, _ = build_datasets(
        catalog, target_idxs, non_target_idx, sidecar.batch_size, rng,
        input_repr=sidecar.model.input_frontend,
    )
    return train_eval_ds, val_ds, test_ds, final_labels


def bake_for_sidecar(
    model: Any, catalog: Any, sidecar: Any, paths: Any,
    *, n_representative: int = 2000,
) -> Any:
    """One-call PTQ + eval + save for a trained model + its sidecar."""

    train_eval_ds, val_ds, test_ds, final_labels = build_eval_datasets(catalog, sidecar)
    return bake_eval_save(
        model, train_ds=train_eval_ds, val_ds=val_ds, test_ds=test_ds,
        label_names=final_labels, paths=paths, train=sidecar,
        n_representative=n_representative,
    )


def spec_label(spec: Any) -> str:
    """Display label used across the plot, prints, and on-disk run dirs."""
    return f"{spec.name}_x{spec.multi:g}"


def train_all(
    specs: list[Any],
    catalog: Any,
    collection: str,
    *,
    hp: Any,
    on_epoch_end: Callable[[Any, int, dict[str, float]], None] | None = None,
    chunk_size: int = 20,
) -> list[tuple[str, Any, Any]]:
    """Co-train every spec on the same raw-audio batches: featurize once per
    unique input_repr per batch, dispatch to every spec's train_on_batch. On
    per-spec EarlyStopping (or after the final epoch), restore best weights, save
    .keras and sidecar, then bake INT8 and write results.json.

    chunk_size caps how many specs co-train at once, needed when 20+ Adam states
    and activations won't fit on the GPU. Datasets are built once and reused
    across chunks; between chunks we drop refs and clear the Keras session so the
    next chunk starts from a clean GPU.

    Returns (label, BakedRun, TrainSidecar) per spec, in finalize order.
    on_epoch_end(spec, epoch, logs) fires once per (spec, epoch).
    """
    import gc
    import numpy as np
    import tensorflow as tf

    from building.polychirp_training.training import TrainSidecar
    from building.polychirp_training.training import SpecState, co_train

    # Resume support, mirroring scale_runner.run_one: skip specs whose
    # results.json already exists so a re-run just fills the gaps.
    skipped = [s for s in specs if s.paths(collection).results_path.exists()]
    specs = [s for s in specs if not s.paths(collection).results_path.exists()]
    for s in skipped:
        print(f"  [{spec_label(s)}] skip, results.json already exists")
    if not specs:
        print("nothing to train: every spec already has results.json")
        return []

    non_target_idx = catalog.class_names.index("non_target")
    target_idxs = [
        i for i, n in enumerate(catalog.class_names) if n != "non_target"
    ]

    # Raw-audio shared batches (input_repr=None): augmentation runs once and all
    # specs see the same clip order; only featurization differs per spec.
    rng = np.random.default_rng(hp.seed)
    train_ds, val_ds, test_ds, final_labels, meta = build_datasets(
        catalog, target_idxs, non_target_idx, hp.batch_size, rng,
        input_repr=None, training_augmentation=hp.augment,
    )
    rng_clean = np.random.default_rng(hp.seed)
    train_eval_ds, _ = build_dataset_from_catalog(
        catalog, target_idxs, non_target_idx, hp.batch_size, rng_clean,
        split="train", input_repr=None, augment=False,
    )

    finalized: list[tuple[str, Any, Any]] = []

    def epoch_hook(state: Any, epoch: int, logs: dict[str, float]) -> None:
        if on_epoch_end is not None:
            on_epoch_end(state.spec, epoch, logs)

    def make_snapshot(
        paths_by_id: dict[int, Any], snapped: list[tuple[Any, Any, Any]]
    ) -> Callable[[Any], None]:
        # Cheap part of finishing a spec: restore best weights, save model and
        # sidecar, collect it. Bake and eval are deferred to one tune_cell pass.
        def snapshot(state: Any) -> None:
            paths = paths_by_id[id(state)]
            if state.best_weights is not None:
                state.model.set_weights(state.best_weights)
            paths.model_path.parent.mkdir(parents=True, exist_ok=True)
            state.model.save(paths.model_path)
            sidecar = TrainSidecar(
                model=state.spec, collection=collection,
                batch_size=hp.batch_size, seed=hp.seed,
                epochs_trained=state.epochs_trained,
            )
            sidecar.save(paths.train_path)
            snapped.append((state.model, paths, sidecar))

        return snapshot

    n_chunks = (len(specs) + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        chunk = specs[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        tf.keras.backend.clear_session()
        gc.collect()
        states = [SpecState(spec=s, model=s.builder(meta.n_classes)) for s in chunk]
        paths_by_id: dict[int, Any] = {
            id(s): s.spec.paths(collection) for s in states
        }
        print(
            f"chunk {chunk_idx + 1}/{n_chunks} ({len(states)} specs): "
            + ", ".join(spec_label(s.spec) for s in states)
        )
        snapped: list[tuple[Any, Any, Any]] = []
        co_train(
            states, train_ds, val_ds, hp=hp,
            on_epoch_end=epoch_hook,
            on_spec_stop=make_snapshot(paths_by_id, snapped),
        )
        # One shared tune pass for the whole chunk: features built once per
        # input_repr, INT8 sweeps parallel across models. Mirrors run_scale.
        baked = tune_cell(
            snapped,
            train_eval_ds=train_eval_ds, val_ds=val_ds, test_ds=test_ds,
            label_names=final_labels,
        )
        for (_, paths, sidecar), run in zip(snapped, baked):
            label = spec_label(sidecar.model)
            print(f"  [{label}] wrote {paths.results_path}")
            finalized.append((label, run, sidecar))
        for s in states:
            s.model = None
            s.best_weights = None
        del states, paths_by_id, snapped

    return finalized


class MultiModelLivePlot:
    """One livelossplot.PlotLosses figure shared across every spec in a co_train
    run. Each model's val curve lands on the same panel per metric. legend=True
    labels each curve by model, readable for a few specs; the default omits it so
    20-spec chunks stay uncluttered."""

    CELL_SIZE = (9, 4)

    def __init__(
        self,
        model_labels: list[str],
        metrics: tuple[str, ...] = ("loss", "accuracy", "precision", "recall"),
        *,
        legend: bool = False,
    ) -> None:
        from livelossplot import PlotLosses
        from livelossplot.outputs import MatplotlibPlot

        groups = {m: [f"{lbl}/val_{m}" for lbl in model_labels] for m in metrics}

        def after_subplot(ax, group_name, x_label):
            ax.set_title(group_name)
            ax.set_xlabel(x_label)
            if legend:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, [l.split("/", 1)[0] for l in labels],
                          fontsize=7, loc="best")
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

        # MatplotlibPlot only; the default ExtremaPrinter spams about 4 lines per
        # spec per epoch (80+ lines for a 20-spec chunk).
        self.plot = PlotLosses(
            groups=groups,
            outputs=[MatplotlibPlot(
                cell_size=self.CELL_SIZE,
                after_subplot=after_subplot,
            )],
        )
        self.step: dict[str, int] = {lbl: -1 for lbl in model_labels}
        self.label_width = max((len(lbl) for lbl in model_labels), default=0)

    def update(self, label: str, logs: dict[str, float]) -> None:
        self.step[label] = self.step.get(label, -1) + 1
        prefixed = {
            f"{label}/{k}": float(v) for k, v in logs.items() if k.startswith("val_")
        }
        self.plot.update(prefixed, current_step=self.step[label])
        self.plot.send()
        t = float(logs.get("loss", 0.0))
        v = float(logs.get("val_loss", 0.0))
        va = float(logs.get("val_accuracy", 0.0))
        vp = float(logs.get("val_precision", 0.0))
        vr = float(logs.get("val_recall", 0.0))
        print(
            f"  {label:<{self.label_width}}  loss {t:.3f}/{v:.3f}  "
            f"acc {va:.3f}  P {vp:.3f}  R {vr:.3f}"
        )


def find_pending_runs(results_dir: Path) -> list[tuple[str, Path, str]]:
    """(label, run_dir, status) for trained runs whose results.json is missing or
    fails to validate against the current RunRecord schema."""
    if not results_dir.exists():
        return []
    pending: list[tuple[str, Path, str]] = []
    for keras_path in sorted(results_dir.rglob("model.keras")):
        run_dir = keras_path.parent
        if not (run_dir / "train.json").exists():
            continue
        rj = run_dir / "results.json"
        if not rj.exists():
            status = "missing"
        else:
            try:
                RunRecord.model_validate_json(rj.read_text())
                continue
            except Exception:
                status = "malformed"
        pending.append((f"{run_dir.parent.name}/{run_dir.name}", run_dir, status))
    return pending


def catch_up_pending_ptq(results_dir: Path, collection: str) -> None:
    """Bake INT8 and write results.json for every pending run under results_dir.
    Datasets are cached by (input_repr, batch_size, seed) so runs that share those
    keys reuse the same tf.data graphs."""
    import tensorflow as tf
    from building.polychirp_training.training import load_dataset_catalog, TrainSidecar
    from building.polychirp_training.paths import RunPaths

    pending = find_pending_runs(results_dir)
    if not pending:
        print("every trained model has a valid results.json")
        return

    print(f"{len(pending)} run(s) need PTQ:")
    for tag, _, status in pending:
        print(f"  - {tag:<48s} [{status}]")

    catalog = load_dataset_catalog(collection)
    ds_cache: dict[tuple[str, int, int], tuple[Any, Any, Any, list[str]]] = {}
    for tag, run_dir, status in pending:
        sidecar = TrainSidecar.load(run_dir / "train.json")
        paths = RunPaths(run_dir)
        key = (str(sidecar.model.input_frontend), sidecar.batch_size, sidecar.seed)
        if key not in ds_cache:
            ds_cache[key] = build_eval_datasets(catalog, sidecar)
        train_eval_ds, val_ds, test_ds, final_labels = ds_cache[key]
        tf.keras.backend.clear_session()
        # load_model restores architecture and weights from the .keras file, which
        # bypasses the auto-numbered layer-name mismatches from a fresh build() +
        # load_weights() pair.
        model = tf.keras.models.load_model(paths.model_path, compile=False)
        print(f"\n=== PTQ {tag} ({status}) ===")
        bake_for_sidecar(model, catalog, sidecar, paths)
        print(f"  wrote {paths.results_path}")

    print("\ndone, re-run the gather cell above to pick up the new results.")


def plot_scaling_panels(
    df: pd.DataFrame, panels: list[tuple[str, str]], suptitle: str
) -> None:
    """Plot the five (column, title) metric panels in a 2x3 grid, hiding the sixth
    axis. Each run is a mean line over k with a +/-std band. Shared by the float
    and quantized scaling views in the scale results notebook."""
    import matplotlib.pyplot as plt

    runs = sorted(df["run"].unique())
    summary = (
        df.groupby(["run", "k"])[[c for c, _ in panels]].agg(["mean", "std"]).sort_index()
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    axes[-1].set_visible(False)
    for ax, (col, title) in zip(axes, panels):
        for run in runs:
            s = summary.loc[run]
            mean, std = s[(col, "mean")], s[(col, "std")].fillna(0.0)
            ax.plot(s.index, mean, marker="o", label=run)
            ax.fill_between(s.index, mean - std, mean + std, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("k (target classes)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontweight="bold")
    plt.tight_layout()
    plt.show()
