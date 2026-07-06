"""2-D CNN on log-mel, global-average head; mel_cnn_new / mel_higher / mel_ultralight differ only by mel frontend."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import (
    MEL_DEFAULT,
    MEL_HIGHER,
    MEL_INPUT_SHAPE,
    MEL_ULTRALIGHT,
    BaseTinyChirpModel,
    compile_model,
)

N_CHANNELS = 20
DENSE_SIZE = 32
KERNEL_SIZE = (3, 3)
SPATIAL_DROPOUT = 0.1
DROPOUT = 0.3


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    channels = max(1, int(round(N_CHANNELS * multiplier)))
    dense_size = max(1, int(round(DENSE_SIZE * multiplier)))
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(channels, KERNEL_SIZE, activation="relu")(inp)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.Conv2D(channels, KERNEL_SIZE, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.Conv2D(channels, KERNEL_SIZE, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(dense_size, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="mel_cnn_new"))


DEFAULT = BaseTinyChirpModel(
    name="mel_cnn_new", builder=build, input_frontend=MEL_DEFAULT, axon=True
)
HIGHER = BaseTinyChirpModel(
    name="mel_higher", builder=build, input_frontend=MEL_HIGHER, axon=True
)
ULTRALIGHT = BaseTinyChirpModel(
    name="mel_ultralight", builder=build, input_frontend=MEL_ULTRALIGHT, axon=True
)
MODELS = [DEFAULT, HIGHER, ULTRALIGHT]
