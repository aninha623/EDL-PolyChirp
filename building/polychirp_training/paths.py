"""Slug and path helpers shared by the build/train/results notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from building.utils import REPO_ROOT

# Artifact filenames inside a run folder. The folder path encodes task/model/size.
MODEL_FILE = "model.keras"
TFLITE_FILE = "model.tflite"
TRAIN_FILE = "train.json"
RESULTS_FILE = "results.json"
NPZ_FILE = "results.npz"


def task_slug(
    target_species: Sequence[str],
    lat: float,
    lon: float,
    radius_km: float,
    *,
    prefix: str = "task",
) -> str:
    """Slug used as the dataset and results folder name.

    Format: prefix_s<n_species>_<lat>_<lon>_r<radius_km>.
    3 species at Paris (48.86, 2.35) with 50 km radius gives task_s3_49_2_r50.
    """
    return (
        f"{prefix}_s{len(target_species)}"
        f"_{round(lat)}_{round(lon)}_r{round(radius_km)}"
    )


def detection_slug(lat: float, lon: float) -> str:
    """Slug for the bird / no-bird detection dataset: detection_<lat>_<lon>."""
    return f"detection_{round(lat)}_{round(lon)}"


def scale_slug(lat: float, lon: float, radius_km: float) -> str:
    """Place slug shared by the dataset and the models/results folder:
    scale_<lat>_<lon>_r<radius_km>."""
    return f"scale_{round(lat)}_{round(lon)}_r{round(radius_km)}"


def size_tag(*multipliers: float) -> str:
    """Multiplier folder name, e.g. size_x4."""
    return "size_x" + "_".join(f"{m:g}" for m in multipliers)


@dataclass(frozen=True)
class RunPaths:
    """Every artifact path for one run, under models/<task>/<model>/size_x<mult>/."""

    dir: Path

    @property
    def model_path(self) -> Path:
        return self.dir / MODEL_FILE

    @property
    def tflite_path(self) -> Path:
        return self.dir / TFLITE_FILE

    @property
    def train_path(self) -> Path:
        return self.dir / TRAIN_FILE

    @property
    def results_path(self) -> Path:
        return self.dir / RESULTS_FILE

    @property
    def npz_path(self) -> Path:
        return self.dir / NPZ_FILE


def task_dir(collection: str, *, root: Path = REPO_ROOT) -> Path:
    """Root holding every model's run folders: models/<collection>/."""
    return root / "models" / collection


def run_dir(collection: str, model: str, *multipliers: float, root: Path = REPO_ROOT) -> Path:
    """Folder holding all artifacts of one run: models/<task>/<model>/size_x<mult>/."""
    return task_dir(collection, root=root) / model / size_tag(*multipliers)


def run_paths(collection: str, model: str, *multipliers: float, root: Path = REPO_ROOT) -> RunPaths:
    """RunPaths for one run."""
    return RunPaths(run_dir(collection, model, *multipliers, root=root))


def discover_runs(
    collection: str, *, marker: str = TFLITE_FILE, root: Path = REPO_ROOT,
) -> list[tuple[str, float]]:
    """Return (model_name, multiplier) for every run that has <marker>.

    Default marker is the baked tflite. Pass e.g. "rust_bench_native.json" to
    discover runs by a different artifact."""
    out: list[tuple[str, float]] = []
    for p in task_dir(collection, root=root).glob(f"*/size_x*/{marker}"):
        tag = p.parts[-2].removeprefix("size_x")
        if "_" in tag:
            continue
        out.append((p.parts[-3], float(tag)))
    return sorted(out)
