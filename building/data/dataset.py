from __future__ import annotations

import random
import re
from pathlib import Path

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

RAW_DATASET_DIR = ROOT / "raw_dataset"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
AUDIOSET_DIR = RAW_DATASET_DIR / "audioset"
BIRDNET_NO_BIRD_DIR = SUBSAMPLES_DIR / "non_target_empty"
DATASETS_DIR = ROOT / "datasets"

SPLIT_NAMES = ("training", "validation", "testing")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 42

# Clips from the same recording must stay in the same split, across all sources:
#   XenoCanto:  [<species>__]XC<id>_<window>.wav        group on XC<id>
#   Macaulay:   [<species>__]ML<id>_<window>.wav        group on ML<id>
#   AudioSet:   <ytid>.wav  or  <ytid>_<offset>.wav     group on YT<ytid>
# YouTube IDs are exactly 11 chars from [A-Za-z0-9_-].
XC_GROUP_RE = re.compile(r"(?:^|__)(XC\d+)_")
ML_GROUP_RE = re.compile(r"(?:^|__)(ML\d+)_")
AS_GROUP_RE = re.compile(r"^([A-Za-z0-9_-]{11})(?:_\d+)?$")


def group_key(clip_path: Path) -> str:
    name = clip_path.name
    m = XC_GROUP_RE.search(name)
    if m:
        return m.group(1)
    m = ML_GROUP_RE.search(name)
    if m:
        return m.group(1)
    m = AS_GROUP_RE.match(clip_path.stem)
    if m:
        return f"YT{m.group(1)}"
    return name


def audioset_round_robin() -> list[Path]:
    pools = [
        sorted(d.glob("*.wav"))
        for d in sorted(AUDIOSET_DIR.iterdir())
        if d.is_dir()
    ] if AUDIOSET_DIR.exists() else []
    pools = [p for p in pools if p]
    if not pools:
        return []
    cursors = [0] * len(pools)
    out: list[Path] = []
    while True:
        progressed = False
        for i, pool in enumerate(pools):
            if cursors[i] < len(pool):
                out.append(pool[cursors[i]])
                cursors[i] += 1
                progressed = True
        if not progressed:
            return out


def balanced_non_target(
    other_species_folders: list[Path],
    per_species_cap: int | None = None,
) -> list[Path]:
    audioset_pool = audioset_round_robin()
    other_pool: list[Path] = []
    for folder in other_species_folders:
        clips = sorted(folder.glob("*.wav"))
        if per_species_cap is not None:
            clips = clips[:per_species_cap]
        other_pool.extend(clips)
    no_bird_pool = (
        sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
        if BIRDNET_NO_BIRD_DIR.exists()
        else []
    )

    n_each = min(len(audioset_pool), len(other_pool))
    out = audioset_pool[:n_each] + other_pool[:n_each] + no_bird_pool
    cap_note = f" cap={per_species_cap}" if per_species_cap is not None else ""
    print(
        f"non_target: {len(out)} = {n_each}/{len(audioset_pool)} audioset "
        f"+ {n_each}/{len(other_pool)} xc_other + {len(no_bird_pool)} no_bird{cap_note}"
    )
    return out


def split_class(clips: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for clip in clips:
        groups.setdefault(group_key(clip), []).append(clip)
    keys = sorted(groups.keys())
    rng.shuffle(keys)

    n = len(clips)
    targets = {
        "training": int(n * SPLIT_RATIOS[0]),
        "validation": int(n * SPLIT_RATIOS[1]),
        "testing": n - int(n * SPLIT_RATIOS[0]) - int(n * SPLIT_RATIOS[1]),
    }
    buckets: dict[str, list[Path]] = {s: [] for s in SPLIT_NAMES}
    for g in keys:
        pick = max(buckets, key=lambda b: targets[b] - len(buckets[b]))
        buckets[pick].extend(groups[g])
    return buckets


def materialize(collection_name: str, class_to_clips: dict[str, list[Path]]) -> Path:
    dataset_root = DATASETS_DIR / collection_name
    for s in SPLIT_NAMES:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SPLIT_SEED)
    for folder, all_clips in class_to_clips.items():
        buckets = split_class(all_clips, rng)
        print(
            f"{folder}: total={len(all_clips)} "
            f"train={len(buckets['training'])} "
            f"val={len(buckets['validation'])} "
            f"test={len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(buckets[split_name]):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())
    return dataset_root


def build_task_dataset(
    collection_name: str,
    target_species: list[str],
    non_target_species: list[str],
    non_target_per_species_cap: int | None = None,
) -> Path:
    class_to_clips: dict[str, list[Path]] = {}
    for name in target_species:
        folder = name.replace(" ", "_")
        class_to_clips[folder] = sorted((SUBSAMPLES_DIR / folder).glob("*.wav"))

    non_target = balanced_non_target(
        [SUBSAMPLES_DIR / n.replace(" ", "_") for n in non_target_species],
        per_species_cap=non_target_per_species_cap,
    )
    if non_target:
        class_to_clips["non_target"] = non_target
    return materialize(collection_name, class_to_clips)


def build_detection_dataset(
    collection_name: str,
    *,
    per_species: int = 500,
    per_provider: int = 250,
) -> Path:
    """Two-class bird vs no-bird collection.

    bird/ pools every species folder (excluding the BirdNET no-bird harvest),
    capped at per_species clips: ideally per_provider XC plus per_provider ML,
    with any shortfall back-filled from the over-represented provider. no_bird/
    is AudioSet round-robin plus the BirdNET no-bird pool.
    """
    bird: list[Path] = []
    species_dirs = sorted(
        d for d in SUBSAMPLES_DIR.iterdir()
        if d.is_dir() and d.name != BIRDNET_NO_BIRD_DIR.name
    )
    for d in species_dirs:
        clips = sorted(d.glob("*.wav"))
        xc = [c for c in clips if XC_GROUP_RE.search(c.name)]
        ml = [c for c in clips if ML_GROUP_RE.search(c.name)]
        picked = xc[:per_provider] + ml[:per_provider]
        deficit = per_species - len(picked)
        if deficit > 0:
            picked.extend((xc[per_provider:] + ml[per_provider:])[:deficit])
        bird.extend(picked[:per_species])

    no_bird = audioset_round_robin() + (
        sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
        if BIRDNET_NO_BIRD_DIR.exists()
        else []
    )
    print(
        f"detection pools: bird={len(bird)} "
        f"({len(species_dirs)} species, cap {per_provider}+{per_provider} per provider)  "
        f"no_bird={len(no_bird)}"
    )
    return materialize(collection_name, {"bird": bird, "no_bird": no_bird})
