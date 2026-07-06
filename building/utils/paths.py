"""Repository paths anchored on pyproject.toml so notebooks at any depth resolve correctly."""

from __future__ import annotations

import pyrootutils

REPO_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
MODELS_DIR = REPO_ROOT / "models"
SRC_DIR = REPO_ROOT / "src"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)
