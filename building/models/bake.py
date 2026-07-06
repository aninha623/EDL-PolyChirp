"""Bake a trained Keras model into an INT8-quantized TFLite flatbuffer.

A Keras model plus a representative dataset become a fully-INT8 .tflite with
int8 I/O. A post-conversion analyzer pulls out the on-flash weight bytes, RAM
arena size and a 4-D-tensor MFLOPs estimate.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras


@contextlib.contextmanager
def silence_native_logs():
    """Redirect fds 1 and 2 to /dev/null so TFLite's C++ logs stay out of the
    cell. contextlib.redirect_* can't catch C-level writes, so dup the real fds."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


def iter_features(dataset: tf.data.Dataset) -> Iterable[np.ndarray]:
    """Yield numpy feature samples (no batch dim) from a (features, label[, sw]) dataset."""
    for elt in dataset.unbatch():
        x = elt[0] if isinstance(elt, tuple) else elt
        yield x.numpy().astype(np.float32)


def build_representative_batches(
    dataset: tf.data.Dataset,
    take: int = 100,
) -> List[np.ndarray]:
    """Collect `take` samples and re-add the batch axis. Any input rank."""
    batches: List[np.ndarray] = []
    for i, x in enumerate(iter_features(dataset)):
        if i >= take:
            break
        batches.append(x[None, ...])
    return batches


def representative_dataset_from_batches(
    batches: Iterable[np.ndarray],
) -> Callable[[], Iterable[List[np.ndarray]]]:
    def gen():
        for sample in batches:
            yield [sample]

    return gen


class TFLiteStats(BaseModel):
    """Footprint numbers extracted from a baked .tflite flatbuffer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    model_size_kb: float
    arena_size_kb: float | None = None
    flops_mflops: float
    input_dtype: str
    output_dtype: str
    input_shape: tuple[int, ...]


_DTYPE_BYTES = {"INT8": 1, "UINT8": 1, "INT16": 2, "INT32": 4, "FLOAT32": 4}


def analyze_tflite(tflite_path: Path) -> TFLiteStats:
    """Run tf.lite.experimental.Analyzer and extract size, arena, flops.

    - model_size_kb: the "Total data buffer size" line (weights plus biases),
      closer to flash usage than file size.
    - arena_size_kb: peak working set, the max over ops of the bytes of their
      live (non-constant) input+output tensors, else None.
    - flops_mflops: coarse proxy, sum of products of 4-D tensor shapes / 1e6.
    """
    tflite_path = Path(tflite_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), silence_native_logs():
        tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_path))
    text = buf.getvalue()

    m = re.search(r"Total data buffer size:\s*(\d+)\s*bytes", text)
    model_size_kb = (
        int(m.group(1)) / 1024.0 if m else tflite_path.stat().st_size / 1024.0
    )

    # Baked int8 graphs are fully static (`shape:`), so the old shape_signature
    # scan never matched. Read every tensor's bytes, flag constants (`RO`), then
    # take the peak over ops of their live input+output tensors.
    tensors: dict[int, tuple[int, bool]] = {}
    for line in text.splitlines():
        m = re.match(r"\s*T#(\d+)\(.*?\)\s*shape(?:_signature)?:\[([-\d,\s]*)\]", line)
        if not m:
            continue
        dims = [max(1, int(d)) for d in m.group(2).split(",") if d.strip()]
        dt = re.search(r"type:(\w+)", line)
        nbytes = int(np.prod(dims)) * _DTYPE_BYTES.get(dt.group(1) if dt else "", 1)
        tensors[int(m.group(1))] = (nbytes, " RO " in line)
    peak = 0
    for line in text.splitlines():
        if not re.match(r"\s*Op#\d+\s", line):
            continue
        live = {int(i) for i in re.findall(r"T#(\d+)", line)}
        peak = max(peak, sum(tensors[i][0] for i in live if i in tensors and not tensors[i][1]))
    arena_size_kb = peak / 1024.0 if peak > 0 else None

    with silence_native_logs():
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
    flops_mflops = sum(
        float(np.prod(t["shape"])) / 1e6
        for t in interp.get_tensor_details()
        if len(t["shape"]) == 4
    )
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return TFLiteStats(
        path=tflite_path,
        model_size_kb=model_size_kb,
        arena_size_kb=arena_size_kb,
        flops_mflops=flops_mflops,
        input_dtype=str(np.dtype(inp["dtype"]).name),
        output_dtype=str(np.dtype(out["dtype"]).name),
        input_shape=tuple(int(s) for s in inp["shape"]),
    )


def bake_model(
    model: "keras.Model",
    rep_dataset: tf.data.Dataset,
    out_path: Path | str,
    *,
    n_representative: int = 100,
    verbose: bool = True,
    denylisted_ops: list[str] | None = None,
    int16_activations: bool = False,
    weight_only: bool = False,
) -> TFLiteStats:
    """Quantize model to INT8 .tflite using rep_dataset for calibration.

    rep_dataset: a non-shuffled slice of train (or val) yielding
    (features, label[, sw]) with features already in the model's input space.
    The first n_representative samples form the calibration set. Default I/O is
    int8, so downstream eval quantizes inputs and dequantizes outputs.

    Three exclusive overrides:
    - denylisted_ops: keep the named ops in float (auto QUANTIZE/DEQUANTIZE
      boundaries), rest INT8. For ops with a broken INT8 kernel, e.g. tf.pow
      with a learned per-channel exponent in PCEN; for LEAF pass ['POW', 'DIV'].
      I/O int8.
    - int16_activations: INT16 activations, INT8 weights. Flash unchanged,
      activations get 65,536 levels so logits that saturate under INT8 recover.
      I/O int16.
    - weight_only: dynamic-range PTQ, INT8 weights, float activations, no
      calibration. Recovers float accuracy for LEAF/PCEN at about 17% larger
      flash than full INT8 (15.7 KB vs 13.2 KB). I/O float32.
    """
    if sum(bool(x) for x in (denylisted_ops, int16_activations, weight_only)) > 1:
        raise ValueError(
            "denylisted_ops, int16_activations, weight_only are mutually exclusive."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_batches = build_representative_batches(rep_dataset, take=n_representative)
    if not rep_batches:
        raise ValueError(
            "Representative dataset is empty, cannot calibrate INT8 quantization."
        )

    sample_shape = rep_batches[0].shape
    expected = tuple(model.input_shape)
    if expected[1:] != sample_shape[1:]:
        raise ValueError(
            f"Representative sample shape {sample_shape} does not match model "
            f"input shape {expected}. Pass the dataset used to train this model."
        )

    with tempfile.TemporaryDirectory() as tmp_dir, silence_native_logs():
        input_signature = [
            tf.TensorSpec(shape=[1] + list(expected[1:]), dtype=tf.float32)
        ]
        model.export(tmp_dir, input_signature=input_signature)

        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        rep_gen = representative_dataset_from_batches(rep_batches)
        if not weight_only:
            converter.representative_dataset = rep_gen

        if weight_only:
            # No representative dataset: weights to INT8, activations stay float.
            out_path.write_bytes(converter.convert())
        elif int16_activations:
            # Ops without an INT16x8 kernel fall back to float via TFLITE_BUILTINS.
            # QuantizationDebugger ignores the INT16x8 op set, so use the bare
            # converter and accept whatever I/O type it produces.
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            out_path.write_bytes(converter.convert())
        elif denylisted_ops:
            # Float fallback must be allowed for the denylisted ops to stay in float.
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            debug_options = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_ops=list(denylisted_ops),
                fully_quantize=True,
            )
            debugger = tf.lite.experimental.QuantizationDebugger(
                converter=converter,
                debug_dataset=rep_gen,
                debug_options=debug_options,
            )
            out_path.write_bytes(debugger.get_nondebug_quantized_model())
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            out_path.write_bytes(converter.convert())

    stats = analyze_tflite(out_path)
    if verbose:
        mode = (
            "weight-only" if weight_only
            else "INT16x8" if int16_activations
            else "INT8 (denylist)" if denylisted_ops
            else "INT8"
        )
        print(f"Baked {mode} TFLite -> {out_path}")
        print(f"  flash (weights)  : {stats.model_size_kb:>8.1f} KB")
        if stats.arena_size_kb is not None:
            print(f"  arena (activ.)   : {stats.arena_size_kb:>8.1f} KB")
        print(f"  est. MFLOPs      : {stats.flops_mflops:>8.3f}")
        print(f"  input shape/dtype: {stats.input_shape} / {stats.input_dtype}")
    return stats
