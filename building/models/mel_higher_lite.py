"""mel_cnn_new with narrowed mid/late convs and a flatten head."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import (
    MEL_HIGHER,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

WIDE_CHANNELS = 20
NARROW_CHANNELS = 8
DENSE_SIZE = 32
KERNEL_SIZE = (3, 3)
SPATIAL_DROPOUT = 0.1
DROPOUT = 0.3


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    wide = max(1, int(round(WIDE_CHANNELS * multiplier)))
    narrow = max(1, int(round(NARROW_CHANNELS * multiplier)))
    dense_size = max(1, int(round(DENSE_SIZE * multiplier)))
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(wide, KERNEL_SIZE, activation="relu")(inp)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.Conv2D(narrow, KERNEL_SIZE, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.Conv2D(narrow, KERNEL_SIZE, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(dense_size, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="mel_higher_lite"))


MODEL = BaseTinyChirpModel(
    name="mel_higher_lite", builder=build, input_frontend=MEL_HIGHER, axon=True
)
