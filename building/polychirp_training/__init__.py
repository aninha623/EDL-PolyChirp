"""Training and results engine.

Modules:
- paths: run folders and slug helpers
- dataset: the dataset.json manifest
- training, runner, results: train drivers, results schema, PTQ, eval IO

Only the path helpers are re-exported here. training/runner/results pull in
TensorFlow, so import those from their submodule.
"""

from __future__ import annotations

from .paths import (
    MODEL_FILE,
    NPZ_FILE,
    RESULTS_FILE,
    TFLITE_FILE,
    TRAIN_FILE,
    RunPaths,
    detection_slug,
    discover_runs,
    run_dir,
    run_paths,
    scale_slug,
    size_tag,
    task_dir,
    task_slug,
)

__all__ = [
    "MODEL_FILE",
    "TFLITE_FILE",
    "TRAIN_FILE",
    "RESULTS_FILE",
    "NPZ_FILE",
    "RunPaths",
    "task_slug",
    "detection_slug",
    "scale_slug",
    "size_tag",
    "task_dir",
    "run_dir",
    "run_paths",
    "discover_runs",
]
