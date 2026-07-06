from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING, Annotated
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from building.models._common import InputRepr, Mel
from building.models._common import TinyChirpModel
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    PlainSerializer,
)
import pyrootutils
import tensorflow as tf
from tqdm.auto import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

FIT_VERBOSE = 2
SHUFFLE_BUFFER_CAP = 256  # per-class; file-level shuffle in dataset.py handles bulk randomness
PREFETCH_BUFFER = tf.data.AUTOTUNE
MEL_DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

MelEntry = tuple[int, np.ndarray, Callable[[tf.Tensor], tf.Tensor]]
MEL_FN_CACHE: dict["Mel", MelEntry] = {}


def featurize(audio: tf.Tensor, input_repr: "InputRepr") -> tf.Tensor:
    """Turn raw audio into the tensor a model with `input_repr` expects.

    Eager callers (the scale runner) reuse a cached tf.function for about 4x
    per-batch speedup. Calls from inside another graph (tf.data.map) build the
    mel filterbank as a fresh constant in the current trace and emit raw ops;
    sharing one eager tf.constant across tf.data traces leaks Keras-3
    SymbolicTensors."""
    from building.models import Mel, Time

    if isinstance(input_repr, Time):
        return tf.expand_dims(audio, -1)
    if not isinstance(input_repr, Mel):
        raise ValueError(f"unknown input_repr {input_repr!r}")
    cfg = input_repr
    entry = MEL_FN_CACHE.get(cfg)
    if entry is None:
        # init_scope escapes any tf.data.map / tf.function trace we were called
        # from, so the cached constants and tf.function are owned by the eager
        # runtime, not by whichever trace got there first.
        with tf.init_scope():
            from building.utils.mel_spec import build_rust_mel_matrix

            fft_bins, mel_np = build_rust_mel_matrix(
                num_mel_bins=cfg.n_bins,
                fft_length_mel=cfg.step_window, frame_length=cfg.step_window,
                lower_edge_hz=cfg.min_freq, upper_edge_hz=cfg.max_freq,
            )
            mel_const = tf.constant(mel_np, dtype=tf.float32)

            @tf.function(reduce_retracing=True)
            def traced(audio: tf.Tensor) -> tf.Tensor:
                with tf.device(MEL_DEVICE):
                    stfts = tf.signal.stft(
                        audio,
                        frame_length=cfg.step_window, frame_step=cfg.step_size,
                        fft_length=cfg.step_window,
                        window_fn=lambda n, dtype: tf.signal.hann_window(n, periodic=False, dtype=dtype),
                    )
                    spec = tf.abs(stfts)[..., :fft_bins]
                    mel = tf.tensordot(spec, mel_const, 1)
                    return tf.expand_dims(tf.math.log(mel + 1e-6), -1)

        entry = (fft_bins, mel_np, traced)
        MEL_FN_CACHE[cfg] = entry
    fft_bins, mel_np, traced = entry
    if not tf.inside_function():
        return traced(audio)
    mel_matrix = tf.constant(mel_np, dtype=tf.float32)
    with tf.device(MEL_DEVICE):
        stfts = tf.signal.stft(
            audio,
            frame_length=cfg.step_window, frame_step=cfg.step_size,
            fft_length=cfg.step_window,
            window_fn=lambda n, dtype: tf.signal.hann_window(n, periodic=False, dtype=dtype),
        )
        spec = tf.abs(stfts)[..., :fft_bins]
        mel = tf.tensordot(spec, mel_matrix, 1)
        return tf.expand_dims(tf.math.log(mel + 1e-6), -1)


def featurize_for_eval(ds: tf.data.Dataset, input_repr: "InputRepr") -> tf.data.Dataset:
    """Map a raw-audio dataset through `featurize` for eval/bake passes.

    Handles both 3-tuple (audio, label, sample_weight) train/val datasets and
    2-tuple (audio, label) test datasets without forcing the caller to branch."""
    nag = tf.autograph.experimental.do_not_convert
    es = ds.element_spec
    if isinstance(es, tuple) and len(es) == 3:
        mapped = ds.map(
            nag(lambda a, y, sw: (featurize(a, input_repr), y, sw)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        mapped = ds.map(
            nag(lambda a, y: (featurize(a, input_repr), y)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return mapped.prefetch(PREFETCH_BUFFER)


def configure_tf_for_long_runs() -> None:
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


configure_tf_for_long_runs()


def round3(v: float) -> float:
    return round(v, 3)


# Drop-in for `float` on JSON-bound metric fields: round on input and again on
# dump(), so a value edited after construction still serializes rounded.
RoundedFloat = Annotated[
    float,
    AfterValidator(round3),
    PlainSerializer(round3, return_type=float),
]


@dataclass
class ClassSplit:
    ds: tf.data.Dataset
    count: int


@dataclass
class ClassEntry:
    name: str
    global_idx: int
    train: ClassSplit
    val: ClassSplit
    test: ClassSplit

class DatasetCatalog(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    class_names: list[str]
    entries: list[ClassEntry]  # parallel to class_names


class MacroBlock(BaseModel):
    precision: RoundedFloat
    recall: RoundedFloat
    f1: RoundedFloat
    f2: RoundedFloat
    accuracy: RoundedFloat | None = None
    auc: RoundedFloat | None = None


class TrainSidecar(BaseModel):
    """Sidecar written next to the saved .keras model.

    Carries what the results notebooks need to rebuild the same model and
    pipeline: the `model` identity plus batch_size/seed."""

    model: TinyChirpModel
    collection: str
    batch_size: int
    seed: int
    epochs_trained: int

    @classmethod
    def load(cls, path: Path | str) -> "TrainSidecar":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
        return p


@dataclass
class DatasetMeta:
    epoch_samples: int
    n_classes: int
    class_weights: dict[int, float] | None = None


@dataclass(frozen=True)
class TrainHP:
    """Fit-loop hyperparameters shared by the task and scaling runners."""

    epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    augment: bool = True


@dataclass
class SpecState:
    """Live state for one co-trained spec inside `co_train`."""

    spec: Any  # TinyChirpModel, typed Any to avoid the forward ref
    model: tf.keras.Model
    best_val_loss: float = float("inf")
    best_weights: list[np.ndarray] | None = None
    epochs_no_improve: int = 0
    lr_epochs_no_improve: int = 0
    epochs_trained: int = 0
    stopped: bool = False


def co_train(
    states: list[SpecState],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    hp: TrainHP,
    on_epoch_end: Callable[[SpecState, int, dict[str, float]], None] | None = None,
    on_spec_stop: Callable[[SpecState], None] | None = None,
    verbose: bool = True,
) -> None:
    """Co-train every state on the shared (audio, label, sample_weight) batches.

    Featurization runs once per unique input_repr per batch, then each state's
    model trains on its featurized view. Best-val-loss weights, EarlyStopping,
    and ReduceLROnPlateau are tracked per state in place.

    Callbacks:
    - on_epoch_end(state, epoch, logs): once per (state, epoch); logs holds
      every compiled metric plus its val_ counterpart.
    - on_spec_stop(state): when a state early-stops or finishes the last epoch.

    verbose prints per-epoch headers and progress bars so a long cell isn't
    silent for minutes.
    """
    import math
    from time import perf_counter

    if verbose:
        from tqdm.auto import tqdm
    else:
        def tqdm(it, **_kw): return it

    def card(ds: tf.data.Dataset) -> int | None:
        n = int(ds.cardinality().numpy())
        return None if n < 0 else n

    train_steps = card(train_ds)
    val_steps = card(val_ds)

    for epoch in range(hp.epochs):
        active = [s for s in states if not s.stopped]
        if not active:
            break
        reprs = {s.spec.input_frontend for s in active}
        t_epoch = perf_counter()
        if verbose:
            tags = ", ".join(f"{s.spec.name}x{s.spec.multi:g}" for s in active)
            print(f"epoch {epoch + 1}/{hp.epochs} | {len(active)} active | {tags}")

        load_time = featurize_time = 0.0
        model_time: dict[int, float] = {id(s): 0.0 for s in active}

        for s in active:
            s.model.reset_metrics()
        bar = tqdm(train_ds, total=train_steps, desc=f"  train e{epoch + 1}", leave=False)
        t_iter = perf_counter()
        for audio, y, sw in bar:
            load_time += perf_counter() - t_iter
            t = perf_counter()
            feats = {r: featurize(audio, r) for r in reprs}
            featurize_time += perf_counter() - t
            for s in active:
                t = perf_counter()
                s.model.train_on_batch(
                    feats[s.spec.input_frontend], y, sample_weight=sw,
                )
                model_time[id(s)] += perf_counter() - t
            t_iter = perf_counter()
        # One sync per (spec, metric) at phase end: train_on_batch accumulates
        # into stateful trackers, get_metrics_result() flattens them to
        # {name: scalar}.
        train_logs: dict[int, dict[str, float]] = {
            id(s): {k: float(v) for k, v in s.model.get_metrics_result().items()}
            for s in active
        }

        for s in active:
            s.model.reset_metrics()
        bar = tqdm(val_ds, total=val_steps, desc=f"  val   e{epoch + 1}", leave=False)
        t_iter = perf_counter()
        for audio, y, sw in bar:
            load_time += perf_counter() - t_iter
            t = perf_counter()
            feats = {r: featurize(audio, r) for r in reprs}
            featurize_time += perf_counter() - t
            for s in active:
                t = perf_counter()
                s.model.test_on_batch(
                    feats[s.spec.input_frontend], y, sample_weight=sw,
                )
                model_time[id(s)] += perf_counter() - t
            t_iter = perf_counter()
        val_logs: dict[int, dict[str, float]] = {
            id(s): {k: float(v) for k, v in s.model.get_metrics_result().items()}
            for s in active
        }

        stopped_now: list[SpecState] = []
        for s in active:
            s.epochs_trained += 1
            tm = train_logs.get(id(s), {})
            vm = val_logs.get(id(s), {})
            logs = {**tm, **{f"val_{k}": v for k, v in vm.items()}}
            vloss = float(logs.get("val_loss", math.inf))
            if vloss < s.best_val_loss - hp.min_delta:
                s.best_val_loss = vloss
                s.best_weights = s.model.get_weights()
                s.epochs_no_improve = 0
                s.lr_epochs_no_improve = 0
            else:
                s.epochs_no_improve += 1
                s.lr_epochs_no_improve += 1
                if s.lr_epochs_no_improve >= hp.lr_patience:
                    lr_var = s.model.optimizer.learning_rate
                    old_lr = float(tf.keras.backend.get_value(lr_var))
                    new_lr = max(old_lr * hp.lr_factor, hp.lr_min)
                    if new_lr < old_lr:
                        try:
                            lr_var.assign(new_lr)
                        except Exception:
                            tf.keras.backend.set_value(lr_var, new_lr)
                    s.lr_epochs_no_improve = 0
            if on_epoch_end is not None:
                on_epoch_end(s, epoch, logs)
            if s.epochs_no_improve >= hp.patience:
                s.stopped = True
                stopped_now.append(s)

        if verbose:
            best = " | ".join(
                f"{s.spec.name}x{s.spec.multi:g}={s.best_val_loss:.3f}"
                + (" [stop]" if s.stopped else "")
                for s in active
            )
            total_model = sum(model_time.values())
            per_spec = " ".join(
                f"{s.spec.name}x{s.spec.multi:g}={model_time[id(s)]:.1f}s"
                for s in active
            )
            print(
                f"  epoch {epoch + 1} done in {perf_counter() - t_epoch:.1f}s | "
                f"load={load_time:.1f}s | mel={featurize_time:.1f}s | "
                f"models={total_model:.1f}s ({per_spec})"
            )
            print(f"  best val_loss: {best}")

        if on_spec_stop is not None:
            for s in stopped_now:
                on_spec_stop(s)

    if on_spec_stop is not None:
        for s in states:
            if s.stopped:
                continue
            s.stopped = True
            on_spec_stop(s)


_READ_BATCH = 1  # one file per batch so a corrupt clip only drops itself


def load_dataset_catalog(collection: str) -> DatasetCatalog:
    from building.utils import SAMPLE_RATE

    keras = tf.keras
    dataset_root = ROOT / "datasets" / collection

    training_dir = dataset_root / "training"
    class_names = sorted(d.name for d in training_dir.iterdir() if d.is_dir())
    cache_root = ROOT / ".cache" / "dataset_cache" / collection
    cache_root.mkdir(parents=True, exist_ok=True)

    def build_class_split(split_dir_name: str, class_name: str) -> ClassSplit:
        split_dir = dataset_root / split_dir_name
        cache_prefix = cache_root / f"{split_dir_name}__{class_name}"
        cache_hot = (cache_prefix.with_suffix(".index")).exists()
        if not cache_hot:
            for stale in cache_root.glob(f"{cache_prefix.name}*"):
                stale.unlink()
        ds_raw = keras.utils.audio_dataset_from_directory(
            split_dir,
            labels="inferred",
            class_names=[class_name],
            sampling_rate=SAMPLE_RATE,
            batch_size=_READ_BATCH,
            shuffle=False,
            verbose=False,
        )
        ds: tf.data.Dataset = (
            ds_raw
            # AudioSet occasionally has corrupt headers; skip rather than abort
            .ignore_errors(log_warning=True)
            .map(
                tf.autograph.experimental.do_not_convert(
                    lambda x, _: tf.squeeze(x, axis=-1)
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .unbatch()
            # Disk cache (one file per split+class): decode+length-fix runs once,
            # subsequent epochs stream from disk instead of holding audio in RAM.
            .cache(str(cache_prefix))
        )
        # ignore_errors() makes cardinality UNKNOWN; count once, then reassert.
        count = sum(1 for _ in ds)
        ds = ds.apply(tf.data.experimental.assert_cardinality(count))
        return ClassSplit(ds=ds, count=count)

    entries: list[ClassEntry] = []
    bar = tqdm(class_names, desc="loading catalog", unit="class")
    for global_idx, class_name in enumerate(bar):
        bar.set_postfix_str(class_name)
        entries.append(
            ClassEntry(
                name=class_name,
                global_idx=global_idx,
                train=build_class_split("training", class_name),
                val=build_class_split("validation", class_name),
                test=build_class_split("testing", class_name),
            )
        )

    return DatasetCatalog(class_names=class_names, entries=entries)


def split_of(
    entry: ClassEntry, split: Literal["train", "val", "test"]
) -> ClassSplit:
    if split == "train":
        return entry.train
    if split == "val":
        return entry.val
    return entry.test


def inverse_freq_weights(
    counts_by_label: dict[int, int], n_classes: int, total_samples: int
) -> dict[int, float]:
    """Inverse-frequency weights w_c = N / (K * n_c), rescaled to mean=1."""
    raw = {
        lbl: total_samples / (n_classes * count)
        for lbl, count in counts_by_label.items()
    }
    mean_w = sum(raw.values()) / len(raw)
    return {lbl: w / mean_w for lbl, w in raw.items()}


def target_vs_other_weights(
    target_counts: dict[int, int], other_label: int, other_count: int
) -> dict[int, float]:
    """Sample weights so non_target carries half the total mass; the other half
    is split inverse-freq across the K target classes. Mean weight = 1.

    Pure inverse-freq drops non_target to about 0.01 when it outnumbers any
    single target class, which kills the "not a target" gradient. A fixed 50/50
    split keeps that signal strong."""
    k = len(target_counts)
    total = sum(target_counts.values()) + other_count
    weights = {
        lbl: 0.5 * total / (k * count) for lbl, count in target_counts.items()
    }
    weights[other_label] = 0.5 * total / other_count
    return weights


def finalize_dataset(
    combined: tf.data.Dataset,
    *,
    class_weights: dict[int, float],
    n_classes: int,
    batch_size: int,
    split: Literal["train", "val", "test"],
    input_repr: "InputRepr | None",
    augment: bool,
    total_samples: int,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    """Shared tail of the dataset builders: augment, batch, featurize, weight.

    build_dataset_from_catalog and build_grouped_dataset_from_catalog differ
    only in how they assemble `combined`; everything downstream is here.
    input_repr=None skips featurization and emits raw audio (B, T) so a custom
    loop can featurize per model and share the augmented batch.
    """
    nag = tf.autograph.experimental.do_not_convert  # autograph breaks on reloaded notebook closures
    do_shuffle = split == "train"
    weighted = split in ("train", "val")

    # augmentation must run before featurize so polarity/shift/noise also affect
    # the downstream mel STFT
    if do_shuffle and augment:
        from building.data.augmentation import augment_tf
        combined = combined.map(
            nag(lambda x, y: (augment_tf(x), y)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    combined = combined.batch(batch_size)
    cw_tensor = tf.constant(
        [class_weights[i] for i in sorted(class_weights)], dtype=tf.float32
    )

    @nag
    def to_features(xb: tf.Tensor, yb: tf.Tensor):
        x = xb if input_repr is None else featurize(xb, input_repr)
        y = tf.one_hot(yb, n_classes, dtype=tf.float32)
        if weighted:
            return x, y, tf.gather(cw_tensor, yb)
        return x, y

    combined = combined.map(
        to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(PREFETCH_BUFFER)
    return combined, DatasetMeta(
        epoch_samples=int(total_samples),
        n_classes=n_classes,
        class_weights=class_weights if weighted else None,
    )


def build_dataset_from_catalog(
    catalog: DatasetCatalog,
    chosen_idxs: list[int],
    non_target_idx: int,
    batch_size: int,
    rng: np.random.Generator,
    split: Literal["train", "val", "test"],
    input_repr: "InputRepr | None",
    augment: bool = False,
    extra_other_idxs: list[int] | None = None,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    extras = list(extra_other_idxs or [])
    other_idxs = [non_target_idx] + extras

    target_class_splits = [split_of(catalog.entries[i], split) for i in chosen_idxs]
    other_class_splits = [split_of(catalog.entries[i], split) for i in other_idxs]
    target_counts = [cs.count for cs in target_class_splits]
    other_counts = [cs.count for cs in other_class_splits]

    if any(c == 0 for c in target_counts):
        bad = [chosen_idxs[i] for i, c in enumerate(target_counts) if c == 0]
        raise ValueError(f"Class(es) {bad} have no samples in split '{split}'")
    if any(c == 0 for c in other_counts):
        bad = [other_idxs[i] for i, c in enumerate(other_counts) if c == 0]
        raise ValueError(
            f"Other-bucket class(es) {bad} have no samples in split '{split}'"
        )

    n_classes = len(chosen_idxs)
    other_label = n_classes
    do_shuffle = split == "train"

    # AutoGraph breaks on reloaded notebook closures
    nag = tf.autograph.experimental.do_not_convert

    counts_by_label: dict[int, int] = {}
    raw_per_class: list[tf.data.Dataset] = []
    for local_label, (cs, count) in enumerate(
        zip(target_class_splits, target_counts)
    ):
        lbl = tf.constant(local_label, dtype=tf.int32)
        raw_per_class.append(cs.ds.map(nag(lambda x, label=lbl: (x, label))))
        counts_by_label[local_label] = count

    olbl = tf.constant(other_label, dtype=tf.int32)
    other_ds = other_class_splits[0].ds.map(nag(lambda x, label=olbl: (x, label)))
    for cs in other_class_splits[1:]:
        other_ds = other_ds.concatenate(
            cs.ds.map(nag(lambda x, label=olbl: (x, label)))
        )
    raw_per_class.append(other_ds)
    counts_by_label[other_label] = sum(other_counts)

    total_samples = sum(counts_by_label.values())

    if do_shuffle:
        # choose_from_datasets with a numpy-shuffled label schedule visits each
        # sample exactly once per epoch at natural class proportions
        per_class_streams: list[tf.data.Dataset] = []
        for lbl_idx, ds_lbl in enumerate(raw_per_class):
            count = counts_by_label[lbl_idx]
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            per_class_streams.append(
                ds_lbl.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                )
            )
        schedule = np.concatenate(
            [
                np.full(counts_by_label[i], i, dtype=np.int64)
                for i in sorted(counts_by_label)
            ]
        )
        np.random.default_rng(int(rng.integers(0, 2**31 - 1))).shuffle(schedule)
        schedule_ds = tf.data.Dataset.from_tensor_slices(schedule)
        combined = tf.data.Dataset.choose_from_datasets(
            per_class_streams, schedule_ds
        )
        # reassert cardinality to silence Keras' "ran out of data" warning
        combined = combined.apply(
            tf.data.experimental.assert_cardinality(int(total_samples))
        )
    else:
        combined = raw_per_class[0]
        for p in raw_per_class[1:]:
            combined = combined.concatenate(p)

    target_counts = {i: counts_by_label[i] for i in range(n_classes)}
    class_weights = target_vs_other_weights(
        target_counts, other_label, counts_by_label[other_label]
    )
    return finalize_dataset(
        combined,
        class_weights=class_weights,
        n_classes=n_classes,
        batch_size=batch_size,
        split=split,
        input_repr=input_repr,
        augment=augment,
        total_samples=total_samples,
    )


def build_datasets(
    catalog: DatasetCatalog,
    chosen_idxs: list[int],
    non_target_idx: int,
    batch_size: int,
    rng: np.random.Generator,
    *,
    input_repr: "InputRepr | None",
    training_augmentation: bool = False,
    extra_other_idxs: list[int] | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str], DatasetMeta]:
    """Build train / val / test datasets, the matching label list, and the
    train split's DatasetMeta in one shot. Augmentation only applies to the
    train split; val and test stay clean so threshold tuning and reporting
    see the same audio as on disk."""
    splits: dict[str, tf.data.Dataset] = {}
    train_meta: DatasetMeta | None = None
    for split in ("train", "val", "test"):
        ds, meta = build_dataset_from_catalog(
            catalog, chosen_idxs, non_target_idx, batch_size, rng,
            split=split, input_repr=input_repr,
            augment=training_augmentation and split == "train",
            extra_other_idxs=extra_other_idxs,
        )
        splits[split] = ds
        if split == "train":
            train_meta = meta
    label_names = [catalog.class_names[i] for i in chosen_idxs]
    assert train_meta is not None
    return splits["train"], splits["val"], splits["test"], label_names, train_meta


def build_grouped_dataset_from_catalog(
    catalog: DatasetCatalog,
    label_groups: list[list[int]],
    batch_size: int,
    rng: np.random.Generator,
    split: Literal["train", "val", "test"],
    input_repr: "InputRepr | None",
    augment: bool = False,
    balance: Literal["natural", "uniform"] = "natural",
) -> tuple[tf.data.Dataset, DatasetMeta]:
    if not label_groups or any(len(g) == 0 for g in label_groups):
        raise ValueError("label_groups must be a non-empty list of non-empty groups")

    n_classes = len(label_groups)
    do_shuffle = split == "train"
    nag = tf.autograph.experimental.do_not_convert

    counts_by_label: dict[int, int] = {}
    raw_per_label: list[tf.data.Dataset] = []
    for lbl_idx, group in enumerate(label_groups):
        splits = [split_of(catalog.entries[i], split) for i in group]
        counts = [s.count for s in splits]
        if any(c == 0 for c in counts):
            bad = [group[i] for i, c in enumerate(counts) if c == 0]
            raise ValueError(
                f"Class(es) {bad} have no samples in split '{split}'"
            )
        lbl = tf.constant(lbl_idx, dtype=tf.int32)
        ds = splits[0].ds.map(nag(lambda x, label=lbl: (x, label)))
        for s in splits[1:]:
            ds = ds.concatenate(
                s.ds.map(nag(lambda x, label=lbl: (x, label)))
            )
        raw_per_label.append(ds)
        counts_by_label[lbl_idx] = sum(counts)

    total_samples = sum(counts_by_label.values())

    if do_shuffle:
        per_class_streams: list[tf.data.Dataset] = []
        for lbl_idx, ds_lbl in enumerate(raw_per_label):
            count = counts_by_label[lbl_idx]
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            per_class_streams.append(
                ds_lbl.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                ).repeat()
            )
        if balance == "uniform":
            weights = [1.0 / n_classes] * n_classes
            # Cap the epoch at "smallest class visited once" so we don't burn
            # compute on duplicated samples from the rare class.
            epoch_samples = n_classes * min(counts_by_label.values())
        else:
            weights = [counts_by_label[i] / total_samples for i in range(n_classes)]
            epoch_samples = total_samples
        combined = tf.data.Dataset.sample_from_datasets(
            per_class_streams,
            weights=weights,
            seed=int(rng.integers(0, 2**31 - 1)),
            stop_on_empty_dataset=False,
        ).take(epoch_samples)
    else:
        epoch_samples = total_samples
        combined = raw_per_label[0]
        for p in raw_per_label[1:]:
            combined = combined.concatenate(p)

    if balance == "uniform" and split == "train":
        class_weights = {i: 1.0 for i in range(n_classes)}
    else:
        class_weights = inverse_freq_weights(counts_by_label, n_classes, total_samples)
    return finalize_dataset(
        combined,
        class_weights=class_weights,
        n_classes=n_classes,
        batch_size=batch_size,
        split=split,
        input_repr=input_repr,
        augment=augment,
        total_samples=epoch_samples,
    )


def collect_predictions(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for batch in ds:
        xb, yb = batch[0], batch[1]
        y_true.append(yb.numpy())
        y_pred.append(model(xb, training=False).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)
