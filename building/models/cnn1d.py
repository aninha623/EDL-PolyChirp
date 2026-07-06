"""TinyChirp CNN-Time on raw waveform."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import TARGET_AUDIO_LEN, TIME, BaseTinyChirpModel, compile_model

CONV1_FILTERS = 4
CONV2_FILTERS = 8
CONV_KERNEL = 3
POOL_SIZE = 2
POOL_STRIDES = 2
DENSE_HIDDEN = 64
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.1


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    multiplier: float = 1.0,
) -> Model:
    c1 = max(1, int(round(CONV1_FILTERS * multiplier)))
    c2 = max(1, int(round(CONV2_FILTERS * multiplier)))
    dense = max(1, int(round(DENSE_HIDDEN * multiplier)))
    inp = layers.Input(shape=(input_len, 1), name="audio")
    # rank-4 NHWC throughout: the baked tflite avoids a rank-3 Reshape microflow can't lower.
    x = layers.Reshape((input_len, 1, 1), name="to_nhwc")(inp)
    x = layers.Conv2D(c1, (CONV_KERNEL, 1), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(POOL_SIZE, 1), strides=(POOL_STRIDES, 1))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.Conv2D(c2, (CONV_KERNEL, 1), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(POOL_SIZE, 1), strides=(POOL_STRIDES, 1))(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    x = layers.AveragePooling2D(pool_size=(x.shape[1], 1), padding="valid")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(dense, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="cnn1d"))


MODEL = BaseTinyChirpModel(name="cnn1d", builder=build, input_frontend=TIME, need_streaming=True, iree=False)
