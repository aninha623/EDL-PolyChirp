"""Scale-grid runner: sweep sample_idx x k x model x multiplier, one results.json per cell."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from tqdm.auto import tqdm

from building.polychirp_training import RunPaths, run_dir
from building.polychirp_training.results import tune_cell
from building.polychirp_training import training as _training
from building.polychirp_training.training import (
    FIT_VERBOSE,
    DatasetCatalog,
    TinyChirpModel,
    TrainSidecar,
    build_dataset_from_catalog,
    build_datasets,
    featurize,
    load_dataset_catalog,
)

def tf_function_names() -> set[str]:
    from tensorflow.python.eager import context

    ctx = context.context()
    return set(ctx.list_function_names()) if hasattr(ctx, "list_function_names") else set()


def evict_tf_functions(keep: set[str]) -> None:
    """Drop the cell's transient tf.data/TFLite graphs; `keep` holds the pooled+catalog ones."""
    from tensorflow.python.eager import context

    ctx = context.context()
    if not hasattr(ctx, "remove_function"):
        return
    for name in tf_function_names() - keep:
        try:
            ctx.remove_function(name)
        except Exception:
            pass


@dataclass
class PoolEntry:
    model: tf.keras.Model
    init_lr: float


# (spec.name, spec.multi, n_classes) -> compiled model reused across cells.
ModelPool = dict[tuple[str, float, int], PoolEntry]


def build_pooled(spec: TinyChirpModel, n_classes: int) -> PoolEntry:
    """Build+compile a model once for reuse across cells; rebuilding per cell leaks ~0.1 GB."""
    model = spec.builder(n_classes)
    init_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    x = np.zeros((2, *model.input_shape[1:]), np.float32)
    y = np.zeros((2, model.output_shape[-1]), np.float32)
    sw = np.ones((2,), np.float32)
    model(x, training=False)
    model.test_on_batch(x, y, sample_weight=sw)
    model.train_on_batch(x, y, sample_weight=sw)
    return PoolEntry(model, init_lr)


def reset_pooled(entry: PoolEntry, spec: TinyChirpModel, n_classes: int) -> None:
    """New random weights for an independent init, zeroed optimizer, original learning rate."""
    entry.model.set_weights(spec.builder(n_classes).get_weights())
    for v in entry.model.optimizer.variables:
        v.assign(tf.zeros_like(v))
    lr = entry.model.optimizer.learning_rate
    try:
        lr.assign(entry.init_lr)
    except Exception:
        tf.keras.backend.set_value(lr, entry.init_lr)


def scale_cell_collection(task_slug: str, k: int, sample_idx: int) -> str:
    """Grid cell as a sub-collection: <scale_slug>/k<k>/sample_<i>."""
    return f"{task_slug}/k{k}/sample_{sample_idx}"


def scale_run_dir(task_slug: str, model: TinyChirpModel, k: int, sample_idx: int) -> Path:
    return run_dir(scale_cell_collection(task_slug, k, sample_idx), model.name, model.multi)


def scale_run_paths(task_slug: str, model: TinyChirpModel, k: int, sample_idx: int) -> RunPaths:
    return RunPaths(scale_run_dir(task_slug, model, k, sample_idx))


class ScaleGridConfig(BaseModel):
    collection: str
    # one entry per trained size
    specs: list[TinyChirpModel]
    epochs: int
    patience: int
    batch_size: int
    seed: int
    augment: bool = True
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    min_delta: float = 1e-3
    n_representative: int = 2000


Pending = list[tuple[TinyChirpModel, RunPaths]]


@dataclass
class CellState:
    """SpecState plus the run paths for one (k, sample) grid cell."""

    spec: TinyChirpModel
    paths: RunPaths
    model: tf.keras.Model
    best_val_loss: float = math.inf
    best_weights: list | None = None
    epochs_no_improve: int = 0
    lr_epochs_no_improve: int = 0
    epochs_trained: int = 0
    stopped: bool = False


def run_scale(
    config: ScaleGridConfig,
    *,
    n_samples: int,
    k_values: Iterable[int],
) -> list[Path]:
    """Train the whole grid in-process; a re-run resumes by skipping cells already on disk."""
    print(f"[scale] loading dataset catalog {config.collection!r} (decodes every clip) ...")
    t0 = perf_counter()
    catalog = load_dataset_catalog(config.collection)
    print(f"[scale] catalog ready in {perf_counter() - t0:.1f}s ({len(catalog.class_names)} classes)")
    task_slug = config.collection
    non_target_idx = catalog.class_names.index("non_target")
    pool = [i for i, n in enumerate(catalog.class_names) if n != "non_target"]
    ks = [k for k in k_values if 1 <= k <= min(10, len(pool))]
    specs = list(config.specs)

    model_pool: ModelPool = {}
    produced: list[Path] = []
    for sample_idx in range(n_samples):
        for k in ks:
            chosen = (
                np.random.default_rng([config.seed, k, sample_idx])
                .choice(pool, k, replace=False)
                .tolist()
            )
            pending = [
                (s, scale_run_paths(task_slug, s, k, sample_idx))
                for s in specs
            ]
            pending = [(s, p) for s, p in pending if not p.results_path.exists()]
            if not pending:
                continue
            train_cell(catalog, config, pending, chosen, non_target_idx, k, sample_idx, model_pool)
            produced.extend(p.results_path for _, p in pending)
    return produced


def make_sidecar(state: CellState, config: ScaleGridConfig) -> TrainSidecar:
    return TrainSidecar(
        model=state.spec, collection=config.collection,
        batch_size=config.batch_size, seed=config.seed,
        epochs_trained=state.epochs_trained,
    )


def snapshot(state: CellState, config: ScaleGridConfig) -> None:
    """Restore best weights, save model + sidecar; bake/eval deferred to finalize_cell."""
    if state.best_weights is not None:
        state.model.set_weights(state.best_weights)
    state.paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    state.model.save(state.paths.model_path)
    make_sidecar(state, config).save(state.paths.train_path)


def finalize_cell(
    states: list[CellState],
    config: ScaleGridConfig,
    train_eval_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    label_names: list[str],
    k: int,
    sample_idx: int,
    chosen: list[int],
) -> float:
    """Tune every snapshotted spec in one shared pass; returns the tuning wall-clock."""
    t0 = perf_counter()
    tune_cell(
        [(s.model, s.paths, make_sidecar(s, config)) for s in states],
        train_eval_ds=train_eval_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        label_names=label_names,
        n_representative=config.n_representative,
        extra_meta={
            "k": k, "sample_idx": sample_idx,
            "chosen_class_indices": chosen, "chosen_classes": label_names[:-1],
        },
        verbose=False,
    )
    return perf_counter() - t0


def train_cell(
    catalog: DatasetCatalog,
    config: ScaleGridConfig,
    pending: Pending,
    chosen: list[int],
    non_target_idx: int,
    k: int,
    sample_idx: int,
    pool: ModelPool,
) -> None:
    gc.collect()
    n_classes = len(chosen)  # multi-label sigmoid head; non_target is all-zeros
    # Reuse one compiled model per (spec, n_classes); build any new ones before fns_before.
    for s, _ in pending:
        key = (s.name, s.multi, n_classes)
        if key not in pool:
            pool[key] = build_pooled(s, n_classes)
    fns_before = tf_function_names()
    states: list[CellState] = []
    train_ds = train_eval_ds = val_ds = test_ds = None
    try:
        # Fresh rng so every spec sees identical clip sampling; only featurization differs.
        seed_key = [config.seed, k, sample_idx]
        train_ds, val_ds, test_ds, _, meta = build_datasets(
            catalog, chosen, non_target_idx, config.batch_size,
            np.random.default_rng(seed_key),
            input_repr=None, training_augmentation=config.augment,
        )
        assert meta.n_classes == n_classes, (meta.n_classes, n_classes)
        # Clean (no-augment) pass for threshold tuning and the train-set eval tables.
        train_eval_ds, _ = build_dataset_from_catalog(
            catalog, chosen, non_target_idx, config.batch_size,
            np.random.default_rng(seed_key),
            split="train", input_repr=None, augment=False,
        )

        states = []
        for s, p in pending:
            entry = pool[(s.name, s.multi, n_classes)]
            reset_pooled(entry, s, n_classes)
            states.append(CellState(spec=s, paths=p, model=entry.model))
        label_names = [catalog.class_names[i] for i in chosen]

        cell_t0 = perf_counter()
        cell_tag = f"[scale] s{sample_idx} k{k}"
        for epoch in range(config.epochs):
            active = [s for s in states if not s.stopped]
            if not active:
                break
            reprs = {s.spec.input_frontend for s in active}

            # Eager ops sync at the Python boundary, so wall-clock splits the stages.
            t_data = t_feat = 0.0
            t_step: dict[int, float] = {id(s): 0.0 for s in active}
            t_train = perf_counter()
            t_batch = perf_counter()
            for audio, y, sw in train_ds:
                t_data += perf_counter() - t_batch
                t0 = perf_counter()
                feats = {r: featurize(audio, r) for r in reprs}
                t_feat += perf_counter() - t0
                for s in active:
                    t0 = perf_counter()
                    s.model.train_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)
                    t_step[id(s)] += perf_counter() - t0
                t_batch = perf_counter()
            train_elapsed = perf_counter() - t_train

            t_val = perf_counter()
            val_losses: dict[int, list[float]] = {id(s): [] for s in active}
            for audio, y, sw in val_ds:
                feats = {r: featurize(audio, r) for r in reprs}
                for s in active:
                    out = s.model.test_on_batch(feats[s.spec.input_frontend], y, sample_weight=sw)
                    val_losses[id(s)].append(float(out[0] if isinstance(out, (list, tuple)) else out))
            val_elapsed = perf_counter() - t_val

            stopped_now: list[CellState] = []
            for s in active:
                s.epochs_trained += 1
                vloss = float(np.mean(val_losses[id(s)])) if val_losses[id(s)] else math.inf
                if vloss < s.best_val_loss - config.min_delta:
                    s.best_val_loss = vloss
                    s.best_weights = s.model.get_weights()
                    s.epochs_no_improve = 0
                    s.lr_epochs_no_improve = 0
                else:
                    s.epochs_no_improve += 1
                    s.lr_epochs_no_improve += 1
                    if s.lr_epochs_no_improve >= config.lr_patience:
                        lr_var = s.model.optimizer.learning_rate
                        old_lr = float(tf.keras.backend.get_value(lr_var))
                        new_lr = max(old_lr * config.lr_factor, config.lr_min)
                        if new_lr < old_lr:
                            try:
                                lr_var.assign(new_lr)
                            except Exception:
                                tf.keras.backend.set_value(lr_var, new_lr)
                        s.lr_epochs_no_improve = 0
                if s.epochs_no_improve >= config.patience:
                    s.stopped = True
                    stopped_now.append(s)

            bests = " ".join(
                f"{s.spec.name}x{s.spec.multi:g}={s.best_val_loss:.4f}"
                + (" [stop]" if s.stopped else "")
                for s in active
            )
            detail = ""
            if FIT_VERBOSE:
                steps = " ".join(
                    f"{s.spec.name}x{s.spec.multi:g}={t_step[id(s)]:.1f}s" for s in active
                )
                detail = f"data={t_data:.1f}s feat={t_feat:.1f}s {steps} "
            tqdm.write(
                f"{cell_tag} e{epoch + 1}/{config.epochs} "
                f"train={train_elapsed:.1f}s val={val_elapsed:.1f}s {detail}| {bests}"
            )

            for s in stopped_now:
                snapshot(s, config)

        train_total = perf_counter() - cell_t0

        # Still-active specs at the last epoch: snapshot like Keras restore_best_weights.
        for s in states:
            if s.stopped:
                continue
            s.stopped = True
            snapshot(s, config)
        tune_total = finalize_cell(
            states, config, train_eval_ds, val_ds, test_ds, label_names, k, sample_idx, chosen
        )
        epochs_ran = max((s.epochs_trained for s in states), default=0)
        tqdm.write(
            f"{cell_tag} done {len(states)} model(s) | {epochs_ran} ep | "
            f"train {train_total / 60:.1f}min | tune {tune_total:.1f}s"
        )
    finally:
        # Null cached step graphs before evicting, else pooled models dispatch to a removed def next cell.
        for s in states:
            if s.model is not None:
                s.model.train_function = None
                s.model.test_function = None
                s.model.predict_function = None
            s.model = None
            s.best_weights = None
        states.clear()
        del train_ds, train_eval_ds, val_ds, test_ds
        _training.MEL_FN_CACHE.clear()
        gc.collect()
        evict_tf_functions(fns_before)
        gc.collect()
