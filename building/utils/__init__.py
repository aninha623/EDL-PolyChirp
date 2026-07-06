"""Public surface of building.utils: re-exports from the focused submodules."""

from __future__ import annotations

from .constants import (
    CLIP_DURATION_SEC,
    FFT_LENGTH_MEL,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HERTZ,
    NUM_MEL_BINS_MEL,
    SAMPLE_RATE,
    SEED,
    TARGET_AUDIO_LEN,
    TARGET_FRAMES_MEL,
    UPPER_EDGE_HERTZ,
)
from .mel_spec import build_rust_mel_matrix, hz_to_mel, mel_to_hz
from .paths import MODELS_DIR, REPO_ROOT, SRC_DIR
from .rust_clips import write_audio_sample_rs

__all__ = [
    # constants
    "SEED",
    "SAMPLE_RATE",
    "CLIP_DURATION_SEC",
    "FRAME_LENGTH",
    "FRAME_STEP",
    "TARGET_AUDIO_LEN",
    "FFT_LENGTH_MEL",
    "NUM_MEL_BINS_MEL",
    "LOWER_EDGE_HERTZ",
    "UPPER_EDGE_HERTZ",
    "TARGET_FRAMES_MEL",
    # paths
    "REPO_ROOT",
    "MODELS_DIR",
    "SRC_DIR",
    # mel_spec
    "hz_to_mel",
    "mel_to_hz",
    "build_rust_mel_matrix",
    # rust_clips
    "write_audio_sample_rs",
]
