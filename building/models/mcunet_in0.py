"""MCUNet-in0 (Lin et al., NeurIPS 2020): the smallest TinyNAS ImageNet net, on log-mel.

Inverted-residual schedule reproduced exactly from the released
mcunet-10fps_imagenet.json (no SE, no feature-mix layer; classifier on 192 ch).
"""

from __future__ import annotations

from keras import Model, layers

from building.models._common import (
    MEL_DEFAULT,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

STEM_FILTERS = 24
# (out_channels, kernel, stride, expand_ratio, residual) per block.
BLOCKS = [
    (16, 3, 1, 1, False),
    (16, 5, 2, 4, False),
    (16, 3, 1, 3, True),
    (16, 3, 1, 4, True),
    (24, 7, 2, 5, False),
    (24, 5, 1, 4, True),
    (24, 5, 1, 5, True),
    (48, 7, 2, 5, False),
    (48, 5, 1, 4, True),
    (56, 3, 1, 5, False),
    (56, 5, 1, 4, True),
    (56, 3, 1, 5, True),
    (112, 3, 2, 6, False),
    (112, 5, 1, 4, True),
    (192, 3, 1, 6, False),
]


def inverted_res_block(x, out_ch: int, kernel: int, stride: int, expand: int, residual: bool, name: str):
    in_ch = x.shape[-1]
    y = x
    if expand != 1:
        y = layers.Conv2D(max(1, round(in_ch * expand)), 1, padding="same", use_bias=False, name=f"{name}_expand")(y)
        y = layers.BatchNormalization(name=f"{name}_expand_bn")(y)
        y = layers.ReLU(6.0, name=f"{name}_expand_relu")(y)
    y = layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(y)
    y = layers.BatchNormalization(name=f"{name}_dw_bn")(y)
    y = layers.ReLU(6.0, name=f"{name}_dw_relu")(y)
    y = layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_project")(y)
    y = layers.BatchNormalization(name=f"{name}_project_bn")(y)
    if residual:
        return layers.Add(name=f"{name}_add")([x, y])
    return y


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    def scale(c: int) -> int:
        return max(1, int(round(c * multiplier)))

    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(scale(STEM_FILTERS), 3, strides=2, padding="same", use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    for i, (out_ch, kernel, stride, expand, residual) in enumerate(BLOCKS):
        x = inverted_res_block(x, scale(out_ch), kernel, stride, expand, residual, name=f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="mcunet_in0"))


MODEL = BaseTinyChirpModel(
    name="mcunet_in0", builder=build, input_frontend=MEL_DEFAULT, microflow=False, axon=False, iree=False
)
