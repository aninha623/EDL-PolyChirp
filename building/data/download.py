"""Per-species recording download plus BirdNET extraction. Idempotent.

The download pipeline is source-agnostic: an audio Source (XenoCanto, eBird/
Macaulay Library, ...) supplies the listing and the per-recording fetch, and
this module drives the BirdNET extraction and non-target bucketing identically
for both.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import os
import random
import sys
import threading
from pathlib import Path

import librosa
import numpy as np
import pyrootutils
import soundfile as sf
from birdnetlib import Recording as BirdNETRecording
from birdnetlib.analyzer import Analyzer as ModelAnalyzer
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .sources import RecordingEntry, Source

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
load_dotenv()

RAW_DATASET_DIR = ROOT / "raw_dataset"
SPECIES_DIR = RAW_DATASET_DIR / "species"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
LISTINGS_DIR = RAW_DATASET_DIR / "listings"
BIRDNET_NO_BIRD_DIR = SUBSAMPLES_DIR / "non_target_empty"

TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 3.0
BIRDNET_THRESHOLD = 0.92
# BirdNET runs at this min_conf; any detection at or above it disqualifies the
# recording from no_bird mining (the 0.1-0.92 band is ambiguous), while only
# detections clearing BIRDNET_THRESHOLD are mined as target clips.
NO_BIRD_FLOOR = 0.1
NO_BIRD_CAP = 20_000
POOL_SIZE = 10

_analyzers: dict[int, ModelAnalyzer] = {}
_analyzers_lock = threading.Lock()
_stderr_lock = threading.Lock()
# Keyed by (species_key, prefix) per source, or plain str for the shared bucket.
_clip_indices: dict[object, int] = {}
_nt_lock = threading.Lock()


@contextlib.contextmanager
def quiet():
    with _stderr_lock:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_out = os.dup(1)
        old_err = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)


def get_analyzer() -> ModelAnalyzer:
    tid = threading.get_ident()
    if tid not in _analyzers:
        with _analyzers_lock:
            if tid not in _analyzers:
                _analyzers[tid] = ModelAnalyzer()
    return _analyzers[tid]


def next_idx(key: object, directory: Path) -> int:
    if key not in _clip_indices:
        _clip_indices[key] = (
            max(
                (
                    int(p.stem.rsplit("_", 1)[1])
                    for p in directory.glob("*.wav")
                    if p.stem.rsplit("_", 1)[-1].isdigit()
                ),
                default=-1,
            )
            + 1
        )
    return _clip_indices[key]


def write_clip(audio_path: Path, start: float, clip_path: Path) -> bool:
    if clip_path.exists():
        return True
    try:
        with quiet():
            sig, _ = librosa.load(
                str(audio_path),
                sr=TARGET_SAMPLE_RATE,
                offset=start,
                duration=CLIP_DURATION,
                mono=True,
                res_type="kaiser_fast",
            )
        target_len = int(CLIP_DURATION * TARGET_SAMPLE_RATE)
        if len(sig) < target_len:
            sig = np.pad(sig, (0, target_len - len(sig)))
        else:
            sig = sig[:target_len]
        sf.write(str(clip_path), sig, TARGET_SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"    skip clip {audio_path.name} @ {start:g}s: {e}")
        return False


async def fetch_listing(source: Source, scientific_name: str) -> list[RecordingEntry]:
    """Listings are cached per (source, species) so we hit the network once."""
    species_key = scientific_name.replace(" ", "_")
    listing_path = LISTINGS_DIR / "available" / source.name / f"{species_key}.csv"
    if listing_path.exists():
        with listing_path.open("r", newline="", encoding="utf-8") as f:
            return [
                RecordingEntry(
                    rec_id=r["rec_id"],
                    filename=r["filename"],
                    file_url=r["file_url"],
                    quality=r.get("quality", ""),
                )
                for r in csv.DictReader(f)
            ]

    rows = await source.fetch_listing(scientific_name)
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    with listing_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rec_id", "filename", "file_url", "quality"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "rec_id": r.rec_id,
                    "filename": r.filename,
                    "file_url": r.file_url,
                    "quality": r.quality,
                }
            )
    return rows


async def download_species_one_source(
    scientific_name: str,
    target_from_source: int,
    source: Source,
    *,
    position_base: int = 0,
    verbose: bool = True,
) -> None:
    """Drive source until species_dir holds target_from_source clips for this
    source's prefix. Idempotent and safe to run concurrently per source. State
    (clip counter, processed.csv) is keyed by prefix and name.
    """
    species_key = scientific_name.replace(" ", "_")
    target_lower = scientific_name.lower()

    species_dir = SUBSAMPLES_DIR / species_key
    raw_dir = SPECIES_DIR / species_key
    for d in (species_dir, raw_dir, BIRDNET_NO_BIRD_DIR):
        d.mkdir(parents=True, exist_ok=True)
    processed_csv = LISTINGS_DIR / "processed" / source.name / f"{species_key}.csv"
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    tag = f"[{scientific_name} / {source.name}]"
    idx_key = (species_key, source.prefix)

    clip_count = sum(1 for _ in species_dir.glob(f"{source.prefix}*.wav"))
    if clip_count >= target_from_source:
        if verbose:
            print(f"{tag} {clip_count}/{target_from_source} clips already on disk, skipping.")
        return

    available = await fetch_listing(source, scientific_name)

    if processed_csv.exists():
        with processed_csv.open("r", newline="", encoding="utf-8") as f:
            processed = {
                r["recording"] for r in csv.DictReader(f) if r.get("recording")
            }
    else:
        processed = set()
    pending = [r for r in available if r.filename not in processed]
    random.shuffle(pending)
    pending = pending[: target_from_source - clip_count]
    if not pending:
        if verbose:
            print(
                f"{tag} no pending recordings "
                f"(available={len(available)}, already processed={len(processed)}, "
                f"on disk={clip_count}/{target_from_source})."
            )
        return

    if verbose:
        print(
            f"{tag} {clip_count}/{target_from_source} on disk; "
            f"queued {len(pending)} of {len(available)} available."
        )

    _clip_indices[idx_key] = clip_count
    species_lock = threading.Lock()

    def handle(entry: RecordingEntry) -> int:
        nonlocal clip_count
        with species_lock:
            if clip_count >= target_from_source:
                return 0

        rec_id = entry.rec_id
        audio_path = raw_dir / entry.filename

        try:
            source.download(entry, audio_path)
        except Exception as e:
            print(f"{tag} download failed {entry.filename}: {e}")
            return 0

        clips_written = 0
        try:
            with quiet():
                rec = BirdNETRecording(
                    get_analyzer(), str(audio_path), min_conf=NO_BIRD_FLOOR
                )
                rec.analyze()

            # Mine clips for detections clearing BIRDNET_THRESHOLD; treat the
            # recording as no_bird only if nothing at all cleared NO_BIRD_FLOOR.
            target_dets = [
                d for d in rec.detections
                if target_lower in d.get("scientific_name", "").strip().lower()
                and float(d.get("confidence", 0.0)) >= BIRDNET_THRESHOLD
            ]
            no_detections = not rec.detections
            del rec

            with species_lock:
                for det in target_dets:
                    if clip_count >= target_from_source:
                        break
                    idx = _clip_indices[idx_key]
                    if write_clip(
                        audio_path,
                        float(det.get("start_time", 0.0)),
                        species_dir / f"{source.prefix}{rec_id}_{idx:05d}.wav",
                    ):
                        _clip_indices[idx_key] += 1
                        clip_count += 1
                        clips_written += 1
                        if clip_count == target_from_source and verbose:
                            print(f"{tag} reached {target_from_source} clips.")

            with _nt_lock:
                if no_detections:
                    nt_key = "non_target_empty"
                    next_idx(nt_key, BIRDNET_NO_BIRD_DIR)
                    if _clip_indices[nt_key] < NO_BIRD_CAP:
                        try:
                            info = sf.info(str(audio_path))
                            max_start = max(
                                0.0, float(info.duration) - CLIP_DURATION
                            )
                            start = (
                                random.uniform(0.0, max_start)
                                if max_start > 0
                                else 0.0
                            )
                        except Exception:
                            start = 0.0
                        out = (
                            BIRDNET_NO_BIRD_DIR
                            / f"{source.prefix}{rec_id}_{_clip_indices[nt_key]:05d}.wav"
                        )
                        if write_clip(audio_path, start, out):
                            _clip_indices[nt_key] += 1

        except Exception as e:
            print(f"{tag} BirdNET failed {entry.filename}: {e}")
        else:
            with species_lock:
                write_header = (
                    not processed_csv.exists() or processed_csv.stat().st_size == 0
                )
                with processed_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f, fieldnames=["recording", "clips_extracted"]
                    )
                    if write_header:
                        w.writeheader()
                    w.writerow(
                        {
                            "recording": entry.filename,
                            "clips_extracted": str(clips_written),
                        }
                    )
        finally:
            audio_path.unlink(missing_ok=True)
        return clips_written

    sem = asyncio.Semaphore(POOL_SIZE)
    dl_lock = asyncio.Lock()
    dl_last = [0.0]
    dl_interval = source.dl_interval
    pbar_recs = tqdm(
        total=len(pending),
        desc=f"[{scientific_name}/{source.name}] recs",
        position=position_base,
        leave=True,
        disable=not verbose,
    )
    pbar_clips = tqdm(
        total=target_from_source,
        initial=clip_count,
        desc=f"[{scientific_name}/{source.name}] clips",
        position=position_base + 1,
        leave=True,
        disable=not verbose,
    )

    async def run(entry: RecordingEntry) -> None:
        if dl_interval > 0:
            async with dl_lock:
                now = asyncio.get_event_loop().time()
                wait = dl_interval - (now - dl_last[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                dl_last[0] = asyncio.get_event_loop().time()
        async with sem:
            written = await asyncio.to_thread(handle, entry)
        pbar_recs.update(1)
        if written:
            pbar_clips.update(written)

    try:
        await asyncio.gather(*[run(r) for r in pending])
    finally:
        pbar_clips.close()
        pbar_recs.close()


async def download_and_process(
    scientific_name: str,
    per_source_clips: int,
    sources: list[Source],
    *,
    verbose: bool = True,
) -> None:
    """Build the species' clip pool from multiple sources.

    Each source has a quota of per_source_clips (global target is
    per_source_clips * len(sources)). Phase A runs all sources in parallel up
    to their quota. If a source runs short, Phase B walks the sources
    sequentially, letting each exceed its quota to cover the shortfall.
    """
    if not sources:
        raise ValueError("download_and_process: need at least one source")

    total_target = per_source_clips * len(sources)
    species_key = scientific_name.replace(" ", "_")
    species_dir = SUBSAMPLES_DIR / species_key
    species_dir.mkdir(parents=True, exist_ok=True)

    def current_total() -> int:
        return sum(1 for _ in species_dir.glob("*.wav"))

    def count_from(src: Source) -> int:
        if not src.prefix:  # empty prefix would match every clip
            return 0
        return sum(1 for _ in species_dir.glob(f"{src.prefix}*.wav"))

    source_list = ", ".join(
        f"{s.name}={count_from(s)}/{per_source_clips}" for s in sources
    )
    start_count = current_total()
    if verbose:
        print(
            f"\n=== [{scientific_name}] target={total_target} clips "
            f"({per_source_clips}/source x {len(sources)} sources)  "
            f"on disk={start_count} [{source_list}] ==="
        )

    if start_count >= total_target:
        if verbose:
            print(f"[{scientific_name}] already at target, nothing to do.")
        return

    async def try_one_source(
        src: Source, target_for_src: int, phase: str, position_base: int = 0
    ) -> None:
        # A per-source failure must not abort the batch; log and move on.
        try:
            await download_species_one_source(
                scientific_name, target_for_src, src,
                position_base=position_base, verbose=verbose,
            )
        except Exception as e:
            print(
                f"[{scientific_name} / {src.name}] {phase} failed: "
                f"{type(e).__name__}: {e}"
            )

    if verbose:
        print(f"[{scientific_name}] --- Phase A: per-source quota (parallel) ---")
    await asyncio.gather(
        *[
            try_one_source(src, per_source_clips, "Phase A", position_base=i * 2)
            for i, src in enumerate(sources)
        ]
    )

    if current_total() < total_target:
        if verbose:
            print(
                f"[{scientific_name}] --- Phase B: top up "
                f"(have {current_total()}/{total_target}) ---"
            )
        for src in sources:
            shortfall = total_target - current_total()
            if shortfall <= 0:
                break
            await try_one_source(src, count_from(src) + shortfall, "Phase B")

    if verbose:
        per_source_summary = ", ".join(f"{s.name}={count_from(s)}" for s in sources)
        print(
            f"[{scientific_name}] done. "
            f"final {current_total()}/{total_target} clips on disk "
            f"[{per_source_summary}]."
        )
