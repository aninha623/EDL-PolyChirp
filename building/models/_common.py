"""Shared model constants, the model-identity types, and the compile helper.

Each model module defines a BaseTinyChirpModel (builder, input frontend, which
runtimes can run it); a TinyChirpModel pins one to a width multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Union

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

if TYPE_CHECKING:
    import keras

    from building.polychirp_training import RunPaths
else:
    import tensorflow as tf

    keras = tf.keras

from keras import Model

from building.utils import (
    NUM_MEL_BINS_MEL as NUM_MEL_BINS,
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TARGET_FRAMES_MEL as TARGET_FRAMES,
)

MEL_INPUT_SHAPE = (TARGET_FRAMES, NUM_MEL_BINS, 1)


def compile_model(model: Model) -> Model:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


@dataclass(frozen=True)
class Time:
    pass


@dataclass(frozen=True)
class Mel:
    n_bins: int
    step_size: int
    step_window: int
    min_freq: float
    max_freq: float

    @property
    def n_frames(self) -> int:
        return 1 + (TARGET_AUDIO_LEN - self.step_window) // self.step_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.n_frames, self.n_bins, 1)


InputRepr = Union[Time, Mel]

TIME = Time()
MEL_DEFAULT = Mel(n_bins=80, step_size=256, step_window=1024, min_freq=80.0, max_freq=8000.0)
MEL_HIGHER = Mel(n_bins=40, step_size=256, step_window=1024, min_freq=2500.0, max_freq=8000.0)
MEL_ULTRALIGHT = Mel(n_bins=40, step_size=480, step_window=512, min_freq=2500.0, max_freq=8000.0)


@dataclass(frozen=True)
class BaseTinyChirpModel:
    """One architecture: Keras builder, the frontend it consumes, which on-device runtimes can run it."""

    name: str
    builder: Callable[..., Model]
    input_frontend: InputRepr
    microflow: bool = True
    # Only the streaming microflow kernel can run this model; skip the normal run.
    need_streaming: bool = False
    axon: bool = False
    iree: bool = True
    # Quantized (INT8) metrics are unusable for this model; omit it from any
    # graph drawn over quantized data.
    quantize_collapse: bool = False


def build_base(base: BaseTinyChirpModel, n_classes: int, multiplier: float = 1.0) -> Model:
    """Invoke a base model's builder with the mel input shape and width."""
    kwargs = {}
    if isinstance(base.input_frontend, Mel):
        kwargs["input_shape"] = base.input_frontend.input_shape
    if multiplier != 1.0:
        kwargs["multiplier"] = multiplier
    return base.builder(n_classes, **kwargs)


class TinyChirpModel(BaseModel):
    """A base model pinned to a width multiplier: the identity threaded through
    training and eval. builder(), input_frontend and paths() derive from it.
    base_model accepts a BaseTinyChirpModel or its name and serializes back to
    that name, so a sidecar is {base_model, multi}.
    """

    model_config = ConfigDict(frozen=True)

    base_model: Any
    multi: float = 1.0

    @field_validator("base_model", mode="before")
    @classmethod
    def resolve_base(cls, v: object) -> BaseTinyChirpModel:
        if isinstance(v, BaseTinyChirpModel):
            return v
        from building.models import base_model_by_name

        return base_model_by_name(v)

    @field_serializer("base_model")
    def serialize_base(self, base: BaseTinyChirpModel) -> str:
        return base.name

    @property
    def name(self) -> str:
        return self.base_model.name

    @property
    def input_frontend(self) -> InputRepr:
        return self.base_model.input_frontend

    def builder(self, n_classes: int) -> Model:
        return build_base(self.base_model, n_classes, self.multi)

    def paths(self, collection: str) -> "RunPaths":
        """All artifact paths for this model under models/<task>/<name>/size_x<mult>/."""
        from building.polychirp_training import run_paths

        return run_paths(collection, self.name, self.multi)
