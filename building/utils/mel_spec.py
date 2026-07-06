"""Rust-compatible mel filterbank + its hz<->mel slope helpers."""

from __future__ import annotations

import numpy as np

from .constants import FRAME_LENGTH, LOWER_EDGE_HERTZ, SAMPLE_RATE, UPPER_EDGE_HERTZ


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_rust_mel_matrix(
    num_mel_bins: int,
    fft_length_mel: int,
    frame_length: int = FRAME_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    lower_edge_hz: float = LOWER_EDGE_HERTZ,
    upper_edge_hz: float = UPPER_EDGE_HERTZ,
) -> tuple[int, np.ndarray]:
    fft_bins = fft_length_mel // 2

    mel_edges = np.zeros(num_mel_bins + 2, dtype=np.int32)
    low_mel = hz_to_mel(lower_edge_hz)
    high_mel = hz_to_mel(upper_edge_hz)
    for i in range(num_mel_bins + 2):
        frac = i / (num_mel_bins + 1)
        mel = low_mel + frac * (high_mel - low_mel)
        hz = mel_to_hz(mel)
        bin_idx = int(((frame_length + 1.0) * hz) / sample_rate)
        mel_edges[i] = min(bin_idx, fft_bins - 1)

    mel_matrix = np.zeros((fft_bins, num_mel_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        left = mel_edges[m]
        center = mel_edges[m + 1]
        right = mel_edges[m + 2]
        for k in range(left, center):
            mel_matrix[k, m] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            mel_matrix[k, m] = (right - k) / max(right - center, 1)

    return fft_bins, mel_matrix
