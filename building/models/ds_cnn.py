"""DS-CNN (MLPerf Tiny KWS, Zhang et al.): stride-2 conv stem + 4 depthwise-separable blocks on log-mel."""

from __future__ import annotations

from math import sqrt

from keras import Model, layers

from building.models._common import (
    MEL_DEFAULT,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

FILTERS = 8
STEM_KERNEL = (10, 4)
STEM_STRIDE = (2, 2)
DW_KERNEL = (3, 3)
NUM_BLOCKS = 4
STEM_DROPOUT = 0.2
HEAD_DROPOUT = 0.4


def separable_block(x, filters: int, name: str):
    x = layers.DepthwiseConv2D(DW_KERNEL, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    return layers.ReLU(name=f"{name}_pw_relu")(x)


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    # Compound scaling, DS-CNN S/M/L style: depth grows with multiplier, width
    # with its sqrt, so total params track multiplier^2 as in the width-only models.
    filters = max(1, int(round(FILTERS * sqrt(multiplier))))
    n_blocks = max(1, int(round(NUM_BLOCKS * multiplier)))
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(filters, STEM_KERNEL, strides=STEM_STRIDE, padding="same", use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.Dropout(STEM_DROPOUT, name="stem_drop")(x)
    for i in range(n_blocks):
        x = separable_block(x, filters, name=f"block{i}")
    x = layers.Dropout(HEAD_DROPOUT, name="head_drop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="ds_cnn"))


MODEL = BaseTinyChirpModel(
    name="ds_cnn", builder=build, input_frontend=MEL_DEFAULT, microflow=True, axon=True, iree=True
)

