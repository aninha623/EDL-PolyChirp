"""SincNet with mel-sized frames (kernel=1024, stride=256) + temporal CNN head."""

from __future__ import annotations

from keras import Model, layers, regularizers

from building.models._common import (
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TIME,
    BaseTinyChirpModel,
    compile_model,
)
from building.models.parts import SincnetConv

NUM_FILTERS = 80
KERNEL_SIZE = 1024
STRIDE = 256
MIN_FREQ = 2000.0
MAX_FREQ = 8000.0
REPULSION_WEIGHT = 1e-2

CONV_FILTERS = 24
CONV_FILTER_SIZE = 8
CONV_STRIDE = 2
DENSE_HIDDEN = 32
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.1
WEIGHT_DECAY = 1e-4


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    multiplier: float = 1.0,
) -> Model:
    conv_filters = max(1, int(round(CONV_FILTERS * multiplier)))
    dense_hidden = max(1, int(round(DENSE_HIDDEN * multiplier)))

    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Reshape((input_len, 1, 1), name="to_nhwc")(inp)

    x = SincnetConv(
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        sample_rate=SAMPLE_RATE,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        repulsion_weight=REPULSION_WEIGHT,
        name="sincnet_convolution",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 1), name="envelope_pool")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(CONV_FILTER_SIZE, 1),
        strides=(CONV_STRIDE, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="temporal_conv_1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 1), name="temporal_pool_1")(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT, name="spatial_dropout_1")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(CONV_FILTER_SIZE, 1),
        strides=(CONV_STRIDE, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="temporal_conv_2",
    )(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT, name="spatial_dropout_2")(x)

    x = layers.AveragePooling2D(
        pool_size=(x.shape[1], 1), padding="valid", name="final_pool"
    )(x)
    x = layers.Flatten(name="flatten")(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT, name="dropout_pre_dense")(x)
    x = layers.Dense(dense_hidden, activation="relu", name="dense_hidden")(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT, name="dropout_post_dense")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="sincnet_large"))


MODEL = BaseTinyChirpModel(name="sincnet_large", builder=build, input_frontend=TIME, iree=False)
