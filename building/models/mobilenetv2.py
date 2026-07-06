"""MobileNetV2 (Sandler et al.) on log-mel; `multiplier` is the width alpha. Faithful to keras.applications."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import (
    MEL_DEFAULT,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

# (expansion, out_channels, stride) per inverted-residual block, ImageNet schedule.
BLOCKS = [
    (1, 16, 1),
    (6, 24, 2), (6, 24, 1),
    (6, 32, 2), (6, 32, 1), (6, 32, 1),
    (6, 64, 2), (6, 64, 1), (6, 64, 1), (6, 64, 1),
    (6, 96, 1), (6, 96, 1), (6, 96, 1),
    (6, 160, 2), (6, 160, 1), (6, 160, 1),
    (6, 320, 1),
]
STEM_FILTERS = 32
LAST_FILTERS = 1280
# multiplier=1 maps to the MCU-fit width alpha (full-width MobileNetV2 is ~2.2 MB).
WIDTH_SCALE = 0.3
BN = dict(epsilon=1e-3, momentum=0.999)


def make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_res_block(x, expansion: int, filters: int, stride: int, alpha: float, name: str):
    in_ch = x.shape[-1]
    out_ch = make_divisible(int(filters * alpha))
    y = x
    if expansion != 1:
        y = layers.Conv2D(expansion * in_ch, 1, padding="same", use_bias=False, name=f"{name}_expand")(y)
        y = layers.BatchNormalization(**BN, name=f"{name}_expand_bn")(y)
        y = layers.ReLU(6.0, name=f"{name}_expand_relu")(y)
    y = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(y)
    y = layers.BatchNormalization(**BN, name=f"{name}_dw_bn")(y)
    y = layers.ReLU(6.0, name=f"{name}_dw_relu")(y)
    y = layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_project")(y)
    y = layers.BatchNormalization(**BN, name=f"{name}_project_bn")(y)
    if in_ch == out_ch and stride == 1:
        return layers.Add(name=f"{name}_add")([x, y])
    return y


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    alpha = multiplier * WIDTH_SCALE
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(make_divisible(STEM_FILTERS * alpha), 3, strides=2, padding="same", use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(**BN, name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    for i, (expansion, filters, stride) in enumerate(BLOCKS):
        x = inverted_res_block(x, expansion, filters, stride, alpha, name=f"block{i}")
    last = make_divisible(LAST_FILTERS * alpha) if alpha > 1.0 else LAST_FILTERS
    x = layers.Conv2D(last, 1, padding="same", use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(**BN, name="head_bn")(x)
    x = layers.ReLU(6.0, name="head_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="mobilenetv2"))


MODEL = BaseTinyChirpModel(
    name="mobilenetv2", builder=build, input_frontend=MEL_DEFAULT, microflow=False, axon=False, iree=False
)
