"""TinyChirp single-head Transformer on raw waveform."""

from __future__ import annotations

import builtins

import keras
import tensorflow as tf
from keras import Model, layers

from building.models._common import TARGET_AUDIO_LEN, TIME, BaseTinyChirpModel, compile_model


keras.config.enable_unsafe_deserialization()

_orig_lambda_cos = layers.Lambda.compute_output_shape

def _lambda_compute_output_shape(self, input_shape):
    try:
        return _orig_lambda_cos(self, input_shape)
    except (NotImplementedError, TypeError):
        if (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple))
        ):
            return tuple(input_shape[0])
        raise

layers.Lambda.compute_output_shape = _lambda_compute_output_shape

CONV_FILTERS = 16
CONV_KERNEL = 3
POOL_SIZE = 2
POOL_STRIDES = 2
N_EMBD = 16
HIDDEN_SIZE = 32
N_LAYERS = 1
DROPOUT = 0.25


def _scaled_dot_product_attention(q, k, v):
    scale = tf.cast(tf.shape(q)[-1], q.dtype) ** -0.5
    scores = tf.matmul(q, k, transpose_b=True) * scale
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, v)


builtins._scaled_dot_product_attention = _scaled_dot_product_attention


def _one_head_attention(x, n_embd: int, name: str):
    q = layers.Dense(n_embd, use_bias=False, name=f"{name}_q")(x)
    k = layers.Dense(n_embd, use_bias=False, name=f"{name}_k")(x)
    v = layers.Dense(n_embd, use_bias=False, name=f"{name}_v")(x)
    attn = layers.Lambda(
        lambda qkv: _scaled_dot_product_attention(*qkv),
        name=f"{name}_sdpa",
    )([q, k, v])
    return layers.Dense(n_embd, name=f"{name}_proj")(attn)


def _transformer_block(x, n_embd: int, hidden_size: int, name: str):
    h = layers.LayerNormalization(name=f"{name}_ln1")(x)
    attn = _one_head_attention(h, n_embd, name=f"{name}_sa")
    x = layers.Add(name=f"{name}_res1")([x, attn])

    h = layers.LayerNormalization(name=f"{name}_ln2")(x)
    h = layers.Dense(hidden_size, activation="relu", name=f"{name}_ff1")(h)
    h = layers.Dense(n_embd, name=f"{name}_ff2")(h)
    return layers.Add(name=f"{name}_res2")([x, h])


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    multiplier: float = 1.0,
) -> Model:
    conv_filters = max(1, int(round(CONV_FILTERS * multiplier)))
    n_embd = max(1, int(round(N_EMBD * multiplier)))
    hidden_size = max(1, int(round(HIDDEN_SIZE * multiplier)))
    inp = layers.Input(shape=(input_len, 1), name="audio")

    x = layers.Conv1D(
        conv_filters, CONV_KERNEL, padding="valid", activation="relu", name="conv1",
    )(inp)
    x = layers.MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDES, name="pool1")(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT, name="dropout1")(x)

    x = layers.GlobalAveragePooling1D(name="adpool")(x)
    x = layers.Reshape((1, conv_filters), name="to_token")(x)

    if conv_filters != n_embd:
        x = layers.Dense(n_embd, name="proj_to_embd")(x)

    for i in range(N_LAYERS):
        x = _transformer_block(x, n_embd, hidden_size, name=f"block{i}")

    x = layers.LayerNormalization(name="ln_f")(x)
    x = layers.Reshape((n_embd,), name="from_token")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="transformer"))


MODEL = BaseTinyChirpModel(
    name="transformer", builder=build, input_frontend=TIME, microflow=False, iree=False
)
