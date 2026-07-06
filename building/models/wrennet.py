"""WrenNet: causal depthwise-separable Conv1D with SE + temporal-attention head on log-mel."""

from __future__ import annotations

import tensorflow as tf
from keras import Model, layers
from keras.saving import register_keras_serializable

from building.models._common import (
    MEL_HIGHER,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

BASE_FILTERS = 32
HIDDEN_DIM = 32
NUM_BLOCKS = 3
SUB_BLOCKS_PER_BLOCK = 2
INITIAL_KERNEL = 5
BLOCK_KERNEL = 3
BLOCK_DILATIONS = ((1, 2), (4, 2), (1, 1))
FINAL_KERNEL_1 = 5
FINAL_DILATION_1 = 2
FINAL_KERNEL_2 = 1
SE_RATIO = 6
DROPOUT = 0.05
BN_MOMENTUM = 0.999
BN_EPS = 1e-3


@register_keras_serializable(package="wrennet")
class TemporalAttentionPool(layers.Layer):
    """Softmax(score(x)) over time, weighted sum to a single context vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1, name="score")

    def call(self, x):
        logits = self.score(x)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(weights * x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def causal_depthwise(x, kernel: int, dilation: int, name: str):
    pad = (kernel - 1) * dilation
    x = layers.ZeroPadding1D((pad, 0), name=f"{name}_pad")(x)
    return layers.DepthwiseConv1D(
        kernel, dilation_rate=dilation, padding="valid", use_bias=False, name=name,
    )(x)


def se_block(x, name: str):
    channels = x.shape[-1]
    reduced = max(1, channels // SE_RATIO)
    s = layers.GlobalAveragePooling1D(keepdims=True, name=f"{name}_squeeze")(x)
    s = layers.Conv1D(reduced, 1, use_bias=False, name=f"{name}_red")(s)
    s = layers.Activation("hard_silu", name=f"{name}_red_act")(s)
    s = layers.Conv1D(channels, 1, use_bias=False, name=f"{name}_exp")(s)
    s = layers.Activation("sigmoid", name=f"{name}_gate")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def phinet_block(x, filters: int, kernel: int, dilation: int, name: str):
    in_c = x.shape[-1]
    # expansion=1 in the paper, so the pointwise expansion keeps width.
    x = layers.Conv1D(in_c, 1, use_bias=False, padding="causal", name=f"{name}_exp")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS, name=f"{name}_exp_bn")(x)
    x = layers.Activation("hard_silu", name=f"{name}_exp_act")(x)
    x = layers.SpatialDropout1D(DROPOUT, name=f"{name}_drop")(x)
    x = causal_depthwise(x, kernel, dilation, name=f"{name}_dw")
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS, name=f"{name}_dw_bn")(x)
    x = layers.Activation("hard_silu", name=f"{name}_dw_act")(x)
    x = se_block(x, name=f"{name}_se")
    x = layers.Conv1D(filters, 1, use_bias=False, padding="causal", name=f"{name}_proj")(x)
    return layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS, name=f"{name}_proj_bn")(x)


def stage(x, filters: int, kernel: int, dilation: int, name: str):
    y = phinet_block(x, filters, kernel, dilation, name=name)
    y = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS, name=f"{name}_bn")(y)
    y = layers.ReLU(name=f"{name}_relu")(y)
    return layers.SpatialDropout1D(DROPOUT, name=f"{name}_outdrop")(y)


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    base = max(1, int(round(BASE_FILTERS * multiplier)))
    hidden = max(1, int(round(HIDDEN_DIM * multiplier)))
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Reshape((input_shape[0], input_shape[1]), name="to_seq")(inp)

    x = stage(x, base, INITIAL_KERNEL, dilation=1, name="init")

    for b in range(NUM_BLOCKS):
        block_in = x
        prev_sub = x
        for s in range(SUB_BLOCKS_PER_BLOCK):
            sub = stage(x, base, BLOCK_KERNEL, BLOCK_DILATIONS[b][s], name=f"b{b}_s{s}")
            x = layers.Add(name=f"b{b}_s{s}_skip")([sub, prev_sub]) if s > 0 else sub
            prev_sub = x
        x = layers.Add(name=f"b{b}_skip")([x, block_in])

    final1 = stage(x, base, FINAL_KERNEL_1, FINAL_DILATION_1, name="final1")
    x = layers.Add(name="final1_skip")([final1, x])
    final2 = stage(x, base, FINAL_KERNEL_2, dilation=1, name="final2")
    x = layers.Add(name="final2_skip")([final2, x])

    x = layers.Dense(hidden, name="proj")(x)
    x = TemporalAttentionPool(name="attn")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="wrennet"))


MODEL = BaseTinyChirpModel(
    name="wrennet", builder=build, input_frontend=MEL_HIGHER, microflow=False, axon=False
)
