"""eBird/Clements taxonomy cache. analysis.py reads the downloaded CSV."""

from __future__ import annotations

from pathlib import Path

import pyrootutils
import requests
from xenocanto3 import Quality

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

# Citation: https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/october-2025/2025-citation-checklist-downloads/
EBIRD_TAXONOMY_URL = (
    "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2025/10/"
    "eBird_taxonomy_v2025.csv"
)
RAW_DATASET_DIR = ROOT / "raw_dataset"
TAXONOMY_FILE = RAW_DATASET_DIR / "taxonomy.csv"

MINIMUM_QUALITY: Quality = Quality.B


def ensure_ebird_taxonomy() -> Path:
    if not TAXONOMY_FILE.exists():
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(EBIRD_TAXONOMY_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        TAXONOMY_FILE.write_text(resp.text, encoding="utf-8")
    return TAXONOMY_FILE


if __name__ == "__main__":
    print(f"eBird taxonomy cached at {ensure_ebird_taxonomy()}")
