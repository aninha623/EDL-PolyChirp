"""TinyChirp NN-Mel: small 2-D CNN on log-mel."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import (
    MEL_DEFAULT,
    MEL_INPUT_SHAPE,
    BaseTinyChirpModel,
    compile_model,
)

N_CHANNELS = 4
HIDDEN_SIZE = 8
DROPOUT = 0.3
CONV_KERNEL_SIZE = (3, 3)
CONV_POOL_SIZE = (2, 2)


def build(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    multiplier: float = 1.0,
) -> Model:
    channels = max(1, int(round(N_CHANNELS * multiplier)))
    hidden = max(1, int(round(HIDDEN_SIZE * multiplier)))
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(channels, CONV_KERNEL_SIZE, activation="relu")(inp)
    x = layers.MaxPooling2D(CONV_POOL_SIZE)(x)
    x = layers.Conv2D(channels, CONV_KERNEL_SIZE, activation="relu")(x)
    x = layers.MaxPooling2D(CONV_POOL_SIZE)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(hidden, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="mel_cnn"))


MODEL = BaseTinyChirpModel(
    name="mel_cnn", builder=build, input_frontend=MEL_DEFAULT, axon=True
)
