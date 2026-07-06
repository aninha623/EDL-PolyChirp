"""SincNet on raw waveform: BN after every conv, ReLU6 activations for INT8 stability."""

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

NUM_FILTERS = 40
KERNEL_SIZE = 64
STRIDE = 16
CONV_FILTERS = 24
CONV_FILTER_SIZE = 8
CONV_STRIDE = 2
DENSE_HIDDEN = 16
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
MIN_FREQ = 2000.0
MAX_FREQ = 8000.0
REPULSION_WEIGHT = 1e-2


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
    x = layers.BatchNormalization(name="sinc_bn")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="envelope_pool")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(CONV_FILTER_SIZE, 1),
        strides=(CONV_STRIDE, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="temporal_conv_1",
    )(x)
    x = layers.BatchNormalization(name="temporal_bn_1")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="temporal_pool_1")(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT, name="spatial_dropout_1")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(CONV_FILTER_SIZE, 1),
        strides=(CONV_STRIDE, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="temporal_conv_2",
    )(x)
    x = layers.BatchNormalization(name="temporal_bn_2")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT, name="spatial_dropout_2")(x)

    x = layers.Flatten(name="flatten")(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT, name="dropout_pre_dense")(x)
    x = layers.Dense(dense_hidden, name="dense_hidden")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.ReLU(max_value=6.0)(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT, name="dropout_post_dense")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="sincnet_norm"))


MODEL = BaseTinyChirpModel(name="sincnet_norm", builder=build, input_frontend=TIME, iree=False)
