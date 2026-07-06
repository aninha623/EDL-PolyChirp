"""Collects the `BaseTinyChirpModel`s each model module defines, keyed by name."""

from __future__ import annotations

from building.models import (
    bake,
    cnn1d,
    ds_cnn,
    leaf,
    mcunet_in0,
    mel_cnn,
    mel_cnn_new,
    mel_higher_lite,
    mobilenetv2,
    model_eval,
    sincnet,
    sincnet_large,
    sincnet_new,
    sincnet_norm,
    squeezenet,
    transformer,
    wrennet,
)
from building.models._common import (
    MEL_DEFAULT,
    MEL_HIGHER,
    MEL_INPUT_SHAPE,
    MEL_ULTRALIGHT,
    NUM_MEL_BINS,
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TARGET_FRAMES,
    TIME,
    BaseTinyChirpModel,
    InputRepr,
    Mel,
    Time,
    TinyChirpModel,
    build_base,
)

BASES: list[BaseTinyChirpModel] = [
    cnn1d.MODEL,
    sincnet.MODEL,
    sincnet_new.MODEL,
    sincnet_norm.MODEL,
    sincnet_large.MODEL,
    *leaf.MODELS,
    mel_cnn.MODEL,
    *mel_cnn_new.MODELS,
    mel_higher_lite.MODEL,
    transformer.MODEL,
    *squeezenet.MODELS,
    wrennet.MODEL,
    ds_cnn.MODEL,
    mobilenetv2.MODEL,
    mcunet_in0.MODEL,
]

REGISTRY: dict[str, BaseTinyChirpModel] = {b.name: b for b in BASES}


def base_model_by_name(name: str) -> BaseTinyChirpModel:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}") from None


# Curated model set shared across notebooks. See MODELS.md.
BASE_LIST: list[TinyChirpModel] = [
    TinyChirpModel(base_model=cnn1d.MODEL),
    TinyChirpModel(base_model=transformer.MODEL),
    TinyChirpModel(base_model=mel_cnn.MODEL),
    TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT),
    TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=2.0),
    TinyChirpModel(base_model=mel_cnn_new.ULTRALIGHT, multi=3.0),
    TinyChirpModel(base_model=sincnet_norm.MODEL),
    TinyChirpModel(base_model=sincnet_norm.MODEL, multi=2.0),
    TinyChirpModel(base_model=wrennet.MODEL),
    TinyChirpModel(base_model=leaf.LEAF_PCEN),
    TinyChirpModel(base_model=ds_cnn.MODEL),
]

# BASE_LIST plus every remaining architecture (mobilenetv2, mcunet, ds_cnn, ...), once each.
EXTENDED_LIST: list[TinyChirpModel] = BASE_LIST + [
    TinyChirpModel(base_model=b)
    for b in BASES
    if b.name not in {m.name for m in BASE_LIST}
]


__all__ = [
    "MEL_INPUT_SHAPE",
    "NUM_MEL_BINS",
    "SAMPLE_RATE",
    "TARGET_AUDIO_LEN",
    "TARGET_FRAMES",
    "REGISTRY",
    "BASES",
    "BASE_LIST",
    "EXTENDED_LIST",
    "Time",
    "Mel",
    "InputRepr",
    "BaseTinyChirpModel",
    "TinyChirpModel",
    "TIME",
    "MEL_DEFAULT",
    "MEL_HIGHER",
    "MEL_ULTRALIGHT",
    "build_base",
    "base_model_by_name",
    "bake",
    "model_eval",
]
