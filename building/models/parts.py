"""Shared layers for the SincNet family."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from keras import layers
from keras.saving import get_custom_objects, register_keras_serializable

from building.models._common import SAMPLE_RATE


@register_keras_serializable()
class SincnetConv(layers.Layer):
    """Learnable band-pass sinc filters. Each filter is a low cutoff f1 plus a
    bandwidth band (both normalised by sample_rate), mel-spaced over
    [min_freq, max_freq] at init. repulsion_weight > 0 penalizes pairwise band
    overlap to push filters apart."""

    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        stride: int,
        sample_rate: int = SAMPLE_RATE,
        min_freq: float = 0.0,
        max_freq: float | None = None,
        repulsion_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = sample_rate / 2.0 if max_freq is None else max_freq
        self.repulsion_weight = repulsion_weight
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

    def build(self, _input_shape):
        mel_points = np.linspace(
            self._hz_to_mel(self.min_freq),
            self._hz_to_mel(self.max_freq),
            self.num_filters + 1,
        )
        hz_points = self._mel_to_hz(mel_points)
        f1_init = hz_points[:-1] / self.sample_rate
        band_init = np.diff(hz_points) / self.sample_rate

        self.f1 = self.add_weight(
            name="f1",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(f1_init),
            trainable=True,
        )
        self.band = self.add_weight(
            name="band",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(band_init),
            trainable=True,
        )
        t = np.linspace(
            -(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size
        )
        self.t = tf.constant(t, dtype=tf.float32)
        window = 0.54 - 0.46 * np.cos(
            2 * math.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )
        self.window = tf.constant(window, dtype=tf.float32)

    def get_filters(self) -> tf.Tensor:
        f1_safe = tf.math.abs(self.f1)
        f2_safe = f1_safe + tf.math.abs(self.band)

        f1_mat = tf.reshape(f1_safe, (1, -1))
        f2_mat = tf.reshape(f2_safe, (1, -1))
        t_mat = tf.reshape(self.t, (-1, 1))

        pi_t = math.pi * t_mat
        denom = tf.where(t_mat == 0.0, 1.0, pi_t)
        filters = (
            tf.math.sin(2.0 * math.pi * f2_mat * t_mat)
            - tf.math.sin(2.0 * math.pi * f1_mat * t_mat)
        ) / denom

        center_values = 2.0 * (f2_mat - f1_mat)
        mask = tf.cast(t_mat == 0.0, tf.float32)
        filters = filters * (1.0 - mask) + center_values * mask

        filters = filters * tf.reshape(self.window, (-1, 1))
        return tf.reshape(filters, (self.kernel_size, 1, self.num_filters))

    def get_filters_nhwc(self) -> tf.Tensor:
        return tf.reshape(
            self.get_filters(), (self.kernel_size, 1, 1, self.num_filters)
        )

    def _repulsion_loss(self) -> tf.Tensor:
        f1 = tf.math.abs(self.f1)
        f2 = f1 + tf.math.abs(self.band)
        max_f1 = tf.maximum(f1[:, None], f1[None, :])
        min_f2 = tf.minimum(f2[:, None], f2[None, :])
        overlap = tf.maximum(min_f2 - max_f1, 0.0)
        overlap = overlap * (1.0 - tf.eye(self.num_filters))
        return self.repulsion_weight * tf.reduce_sum(overlap)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if training and self.repulsion_weight > 0:
            self.add_loss(self._repulsion_loss())
        return tf.nn.conv2d(
            inputs,
            self.get_filters_nhwc(),
            strides=[1, self.stride, 1, 1],
            padding="VALID",
            data_format="NHWC",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "sample_rate": self.sample_rate,
                "min_freq": self.min_freq,
                "max_freq": self.max_freq,
                "repulsion_weight": self.repulsion_weight,
            }
        )
        return config

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# Older .keras saves embed registered_name 'SincnetConv', but
# register_keras_serializable() stores under Custom>SincnetConv. The bare alias
# lets load_model resolve either form without custom_objects.
get_custom_objects()["SincnetConv"] = SincnetConv
