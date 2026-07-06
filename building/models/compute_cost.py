"""Theoretical compute (MACs) and arena RAM per architecture.

Complement to bake.py (baked tflite footprint). Walks a Keras model
pre-quantization and computes:

- total_macs: MACs in Conv1D, Conv2D, Dense, SincnetConv. Activations, BN,
  pool, dropout, reshape ignored.
- arena_elems: max over consecutive layer-output pairs of size(prev) +
  size(next). Times 4 (float32) or 1 (int8) for bytes.
- arena_elems_streaming: line-buffered streaming. Every conv/pool drops its
  full output and contributes a ((kh - 1) * W + kw) * C_in line buffer; 1D
  (3D input) collapses to W=1, giving k * C_in. Element-wise layers pass
  through. Global pool, Flatten, Dense are not streamable.

`baked_rows` joins this streaming arena to the real footprint of each model's
baked int8 tflite (flash, arena, MFLOPs from bake.analyze_tflite).
"""

from __future__ import annotations

from math import floor, log10, prod
from typing import TYPE_CHECKING

from keras import layers
from pydantic import BaseModel

from building.models import build_base
from building.models.bake import analyze_tflite
from building.models.leaf import GaborConv1D, GaussianPool1D
from building.models.parts import SincnetConv

if TYPE_CHECKING:
    from keras import Model

    from building.models import TinyChirpModel


STREAMABLE_PASSTHROUGH = (
    layers.BatchNormalization,
    layers.LayerNormalization,
    layers.Activation,
    layers.ReLU,
    layers.Dropout,
    layers.SpatialDropout1D,
    layers.SpatialDropout2D,
    layers.Reshape,
)

TIME_JOIN = (
    layers.Flatten,
    layers.GlobalAveragePooling1D,
    layers.GlobalAveragePooling2D,
    layers.GlobalMaxPooling1D,
    layers.GlobalMaxPooling2D,
)


class LayerCost(BaseModel):
    name: str
    kind: str
    macs: int
    out_elems: int
    in_channels: int
    kernel_h: int
    kernel_w: int
    streamable: bool
    ring_buffer: int


class ModelCost(BaseModel):
    name: str
    layers: list[LayerCost]
    total_macs: int
    arena_elems: int
    arena_elems_streaming: int


class BakedRow(BaseModel):
    model: str
    multi: float
    mflops: float
    flash_kb: float
    arena_kb: float | None
    arena_kb_streaming: float


def bytes_of(elems: int, dtype: str) -> int:
    return elems * (4 if dtype == "float" else 1)


def shape_elems(shape) -> int:
    return prod(1 if d is None else int(d) for d in shape[1:])


def kernel_tuple(k) -> tuple[int, ...]:
    return tuple(k) if isinstance(k, (tuple, list)) else (int(k),)


def line_ring(kh: int, kw: int, in_shape, in_channels: int) -> int:
    # (kh - 1) previous lines plus kw cells of the current line, per channel.
    # 1D inputs (3D shape) collapse to W=1, giving the kh * C_in ring.
    w = 1 if len(in_shape) < 4 or in_shape[2] is None else int(in_shape[2])
    return ((kh - 1) * w + kw) * in_channels


def conv_cost(name, kind, kh, kw, cout, in_shape, in_channels, out_elems) -> LayerCost:
    spatial = out_elems // max(1, cout)
    return LayerCost(
        name=name, kind=kind,
        macs=kh * kw * in_channels * cout * spatial,
        out_elems=out_elems, in_channels=in_channels,
        kernel_h=kh, kernel_w=kw,
        streamable=True,
        ring_buffer=line_ring(kh, kw, in_shape, in_channels),
    )


def layer_cost(layer, in_shape, out_shape) -> LayerCost:
    name = layer.name
    out_elems = shape_elems(out_shape)
    in_channels = 1 if in_shape[-1] is None else int(in_shape[-1])

    if isinstance(layer, GaborConv1D):
        # Conv1D over (T, 1) to 2*num_filters channels, magnitude-reduced to
        # num_filters. Count the conv MACs.
        k = layer.kernel_size
        cout = 2 * layer.num_filters
        spatial_out = out_elems // max(1, layer.num_filters)
        return LayerCost(
            name=name, kind="gabor", macs=k * in_channels * cout * spatial_out,
            out_elems=out_elems, in_channels=in_channels,
            kernel_h=k, kernel_w=1,
            streamable=True, ring_buffer=line_ring(k, 1, in_shape, in_channels),
        )

    if isinstance(layer, GaussianPool1D):
        # Depthwise smoothing conv: each channel filtered independently.
        k = layer.pool_size
        spatial_out = out_elems // max(1, in_channels)
        return LayerCost(
            name=name, kind="gauss_pool", macs=k * in_channels * spatial_out,
            out_elems=out_elems, in_channels=in_channels,
            kernel_h=k, kernel_w=1,
            streamable=True, ring_buffer=line_ring(k, 1, in_shape, in_channels),
        )

    if isinstance(layer, SincnetConv):
        # Baked SincnetConv is just a Conv2D with a (k, 1) kernel.
        return conv_cost(name, "sincnet", layer.kernel_size, 1, layer.num_filters,
                         in_shape, in_channels, out_elems)

    if isinstance(layer, layers.Conv1D):
        k = kernel_tuple(layer.kernel_size)[0]
        return conv_cost(name, "conv1d", k, 1, layer.filters,
                         in_shape, in_channels, out_elems)

    if isinstance(layer, layers.Conv2D):
        kh, kw = kernel_tuple(layer.kernel_size)
        return conv_cost(name, "conv2d", kh, kw, layer.filters,
                         in_shape, in_channels, out_elems)

    if isinstance(layer, layers.Dense):
        din = in_channels
        dout = layer.units
        # Dense applies to the last axis; multiply by any leading non-batch dims.
        leading = shape_elems(in_shape) // max(1, din)
        return LayerCost(
            name=name, kind="dense", macs=din * dout * leading,
            out_elems=out_elems, in_channels=in_channels,
            kernel_h=0, kernel_w=0,
            streamable=False, ring_buffer=0,
        )

    if isinstance(layer, (layers.MaxPooling1D, layers.AveragePooling1D)):
        k = kernel_tuple(layer.pool_size)[0]
        return LayerCost(
            name=name, kind="pool1d", macs=0, out_elems=out_elems,
            in_channels=in_channels, kernel_h=k, kernel_w=1,
            streamable=True, ring_buffer=line_ring(k, 1, in_shape, in_channels),
        )

    if isinstance(layer, (layers.MaxPooling2D, layers.AveragePooling2D)):
        ph, pw = kernel_tuple(layer.pool_size)
        return LayerCost(
            name=name, kind="pool2d", macs=0, out_elems=out_elems,
            in_channels=in_channels, kernel_h=ph, kernel_w=pw,
            streamable=True, ring_buffer=line_ring(ph, pw, in_shape, in_channels),
        )

    if isinstance(layer, TIME_JOIN):
        return LayerCost(
            name=name, kind="join", macs=0, out_elems=out_elems,
            in_channels=in_channels, kernel_h=0, kernel_w=0,
            streamable=False, ring_buffer=0,
        )

    if isinstance(layer, STREAMABLE_PASSTHROUGH):
        return LayerCost(
            name=name, kind="passthrough", macs=0, out_elems=out_elems,
            in_channels=in_channels, kernel_h=0, kernel_w=0,
            streamable=True, ring_buffer=0,
        )

    return LayerCost(
        name=name, kind="other", macs=0, out_elems=out_elems,
        in_channels=in_channels, kernel_h=0, kernel_w=0,
        streamable=False, ring_buffer=0,
    )


def analyze(model: "Model") -> ModelCost:
    costs: list[LayerCost] = []
    input_layer = model.layers[0]
    input_shape = input_layer.output.shape
    input_elems = shape_elems(input_shape)

    for layer in model.layers[1:]:
        try:
            in_shape = layer.input.shape
            out_shape = layer.output.shape
        except AttributeError:
            # Multi-input layer (Add, Concatenate, Lambda-on-list): use the first.
            in_node = layer.input
            in_shape = in_node[0].shape if isinstance(in_node, list) else in_node.shape
            out_shape = layer.output.shape
        costs.append(layer_cost(layer, in_shape, out_shape))

    total_macs = sum(c.macs for c in costs)

    tensors = [input_elems] + [c.out_elems for c in costs]
    arena_elems = max(tensors[i] + tensors[i + 1] for i in range(len(tensors) - 1))

    # Streaming arena: drop tensors produced by streamable layers, sum ring
    # buffers, take max consecutive pair of the rest. The input tensor is
    # considered streamed iff the first layer is streamable.
    ring_total = sum(c.ring_buffer for c in costs)
    ns_tensors: list[int] = []
    if not costs[0].streamable:
        ns_tensors.append(input_elems)
    for c in costs:
        if not c.streamable:
            ns_tensors.append(c.out_elems)
    if len(ns_tensors) >= 2:
        max_pair = max(ns_tensors[i] + ns_tensors[i + 1] for i in range(len(ns_tensors) - 1))
    elif len(ns_tensors) == 1:
        max_pair = ns_tensors[0]
    else:
        max_pair = 0
    arena_elems_streaming = max_pair + ring_total

    return ModelCost(
        name=model.name,
        layers=costs,
        total_macs=total_macs,
        arena_elems=arena_elems,
        arena_elems_streaming=arena_elems_streaming,
    )


def sig(x: float, n: int = 3) -> float:
    """Round to n significant figures; return int when nothing fractional is left."""
    if x == 0:
        return 0
    digits = n - int(floor(log10(abs(x)))) - 1
    if digits <= 0:
        return int(round(x, digits))
    return round(x, digits)


def baked_rows(
    specs: list["TinyChirpModel"], *, task: str = "bench", n_classes: int = 2
) -> list[BakedRow]:
    """Real footprint of each spec's baked int8 tflite (flash, arena, MFLOPs)
    joined to its theoretical streaming line-buffer arena. n_classes defaults to
    the 2-label bench bake so the streaming graph matches what was baked."""
    rows = []
    for spec in specs:
        stats = analyze_tflite(spec.paths(task).tflite_path)
        cost = analyze(build_base(spec.base_model, n_classes, spec.multi))
        rows.append(BakedRow(
            model=spec.name,
            multi=spec.multi,
            mflops=sig(stats.flops_mflops),
            flash_kb=sig(stats.model_size_kb),
            arena_kb=None if stats.arena_size_kb is None else sig(stats.arena_size_kb),
            arena_kb_streaming=sig(bytes_of(cost.arena_elems_streaming, "int") / 1024),
        ))
    return rows
