"""LEAF Gabor + Gaussian-pool frontend on raw waveform; leaf_pcen (learnable PCEN), leaf_fixed_pcen (frozen PCEN), leaf_log (log compression)."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from keras import Model, layers
from keras.saving import register_keras_serializable

from building.models._common import (
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TIME,
    BaseTinyChirpModel,
    compile_model,
)

LEAF_FILTERS = 40
CONV_FILTERS = 16
KERNEL_SIZE = 64
GABOR_STRIDE = 16
POOL_STRIDE = 4  # GABOR_STRIDE * POOL_STRIDE = 64, preserves the original frame rate.
PCEN_SMOOTH_SIZE = 15
MIN_FREQ = 2000.0
MAX_FREQ = 8000.0
EPS = 1e-3


@register_keras_serializable(package="leaf")
class GaborConv1D(layers.Layer):
    """Complex Gabor filterbank; mel-spaced init, returns per-filter magnitude."""

    def __init__(
        self,
        num_filters,
        kernel_size,
        stride=1,
        sample_rate=SAMPLE_RATE,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

        mel_lo = 2595.0 * np.log10(1.0 + min_freq / 700.0)
        mel_hi = 2595.0 * np.log10(1.0 + max_freq / 700.0)
        edges = 700.0 * (
            10.0 ** (np.linspace(mel_lo, mel_hi, num_filters + 1) / 2595.0) - 1.0
        )
        center_init = ((edges[:-1] + edges[1:]) / 2.0 / sample_rate).reshape(1, 1, -1)
        # Envelope exp(-0.5 (t*b)^2) with t in samples has frequency std
        # sample_rate*b/(2*pi) Hz, so a mel band of width df maps to
        # b = 2*pi*df/sample_rate.
        band_init = (2.0 * np.pi * np.diff(edges) / sample_rate).reshape(1, 1, -1)

        self.center_freqs = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(center_init.astype(np.float32)),
        )
        self.bandwidths = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(band_init.astype(np.float32)),
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
        }

    def get_filters(self):
        limit = (self.kernel_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.kernel_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        env = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        cos_mod = tf.cos(2.0 * math.pi * self.center_freqs * t)
        sin_mod = tf.sin(2.0 * math.pi * self.center_freqs * t)
        return tf.concat([env * cos_mod, env * sin_mod], axis=-1)

    def call(self, inputs):
        conv = tf.nn.conv1d(inputs, self.get_filters(), stride=self.stride, padding="SAME")
        real, imag = tf.split(conv, 2, axis=-1)
        return tf.sqrt(tf.square(real) + tf.square(imag) + EPS)


@register_keras_serializable(package="leaf")
class GaussianPool1D(layers.Layer):
    def __init__(self, num_filters, pool_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.stride = stride
        self.bandwidths = self.add_weight(
            shape=(1, num_filters, 1), initializer=tf.constant_initializer(0.4)
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "pool_size": self.pool_size,
            "stride": self.stride,
        }

    def get_filters(self):
        limit = (self.pool_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.pool_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        return gauss / tf.reduce_sum(gauss, axis=0, keepdims=True)

    def call(self, inputs):
        # GPU depthwise_conv2d needs equal row/col strides; emulate stride=S
        # SAME via explicit pad, stride=1 VALID, then decimate.
        k, s = self.pool_size, self.stride
        t = tf.shape(inputs)[1]
        out_t = -(-t // s)  # ceil(t / s)
        total_pad = tf.maximum((out_t - 1) * s + k - t, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        padded = tf.pad(inputs, [[0, 0], [pad_left, pad_right], [0, 0]])
        smoothed = tf.nn.depthwise_conv2d(
            tf.expand_dims(padded, axis=1),
            tf.expand_dims(self.get_filters(), axis=0),
            strides=[1, 1, 1, 1],
            padding="VALID",
        )[:, 0, :, :]
        if s == 1:
            return smoothed
        # pad to a multiple of s so the strided decimation tosa-reshapes cleanly
        length = smoothed.shape[1]
        if length is not None and length % s:
            smoothed = tf.pad(smoothed, [[0, 0], [0, s - length % s], [0, 0]])
        return smoothed[:, ::s, :]


@register_keras_serializable(package="leaf")
class PCEN(layers.Layer):
    """Per-Channel Energy Normalization: y = ((x+eps)/(eps+M)^alpha + delta)^r - delta^r.

    M = depthwise Gaussian smoothing of x (parallel, INT8-friendly, no IIR scan).
    Replaces log compression; activation distribution is closer to Gaussian.
    Build with trainable=False for the fixed-PCEN variant: parameters stay at
    their per-channel-uniform init, removing the per-channel range divergence.
    """

    def __init__(self, num_filters, smooth_size=PCEN_SMOOTH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.smooth_size = smooth_size

        self.alpha = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(0.96),
            name="alpha",
        )
        self.delta = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(2.0),
            name="delta",
        )
        self.root = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(0.5),
            name="root",
        )
        self.smooth_bw = self.add_weight(
            shape=(1, num_filters, 1),
            initializer=tf.constant_initializer(0.2),
            name="smooth_bw",
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "smooth_size": self.smooth_size,
        }

    def get_smoothing_filters(self):
        limit = (self.smooth_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.smooth_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.smooth_bw))
        return gauss / (tf.reduce_sum(gauss, axis=0, keepdims=True) + 1e-12)

    def call(self, inputs):
        m = tf.nn.depthwise_conv2d(
            tf.expand_dims(inputs, axis=1),
            tf.expand_dims(self.get_smoothing_filters(), axis=0),
            strides=[1, 1, 1, 1],
            padding="SAME",
        )[:, 0, :, :]
        alpha = tf.clip_by_value(self.alpha, 0.0, 1.0)
        root = tf.clip_by_value(self.root, 1e-2, 1.0)
        delta = tf.maximum(self.delta, 0.0)
        smooth = tf.pow(EPS + m, alpha)
        return tf.pow((inputs + EPS) / smooth + delta, root) - tf.pow(delta, root)


@register_keras_serializable(package="leaf")
class LogCompression(layers.Layer):
    """Static log compression, the PTQ-friendly stand-in for PCEN."""

    def call(self, inputs):
        return tf.math.log(inputs + EPS)


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    multiplier: float = 1.0,
    compression: str = "pcen",
) -> Model:
    inputs = layers.Input(shape=(input_len, 1), name="audio")
    x = GaborConv1D(
        num_filters=LEAF_FILTERS,
        kernel_size=KERNEL_SIZE,
        stride=GABOR_STRIDE,
        name="gabor_conv",
    )(inputs)
    x = GaussianPool1D(
        num_filters=LEAF_FILTERS,
        pool_size=KERNEL_SIZE,
        stride=POOL_STRIDE,
        name="gauss_pool",
    )(x)
    if compression == "log":
        x = LogCompression(name="log")(x)
    else:
        x = PCEN(
            num_filters=LEAF_FILTERS,
            name="pcen",
            trainable=compression == "pcen",
        )(x)
    x = layers.BatchNormalization(momentum=0.9, name="compression_bn")(x)

    conv_filters = max(1, int(round(CONV_FILTERS * multiplier)))
    x = layers.Conv1D(filters=conv_filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(filters=conv_filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(conv_filters)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    name = {"pcen": "leaf_pcen", "fixed_pcen": "leaf_fixed_pcen", "log": "leaf_log"}[
        compression
    ]
    return compile_model(Model(inputs, outputs, name=name))


def leaf_pcen(
    n_classes: int, input_len: int = TARGET_AUDIO_LEN, multiplier: float = 1.0
) -> Model:
    return build(n_classes, input_len, multiplier, compression="pcen")


def leaf_fixed_pcen(
    n_classes: int, input_len: int = TARGET_AUDIO_LEN, multiplier: float = 1.0
) -> Model:
    return build(n_classes, input_len, multiplier, compression="fixed_pcen")


def leaf_log(
    n_classes: int, input_len: int = TARGET_AUDIO_LEN, multiplier: float = 1.0
) -> Model:
    return build(n_classes, input_len, multiplier, compression="log")


LEAF_PCEN = BaseTinyChirpModel(
    name="leaf_pcen", builder=leaf_pcen, input_frontend=TIME, microflow=False, iree=False,
    quantize_collapse=True,  # PCEN / log etc collapse
)
LEAF_FIXED_PCEN = BaseTinyChirpModel(
    name="leaf_fixed_pcen", builder=leaf_fixed_pcen, input_frontend=TIME, microflow=False, iree=False,
    quantize_collapse=True,  # PCEN / log etc collapse
)
LEAF_LOG = BaseTinyChirpModel(
    name="leaf_log", builder=leaf_log, input_frontend=TIME, microflow=False, iree=False,
    quantize_collapse=True,  # PCEN / log etc collapse
)
MODELS = [LEAF_PCEN, LEAF_FIXED_PCEN, LEAF_LOG]
