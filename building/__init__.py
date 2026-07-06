"""Quiet TensorFlow's import-time logging before any submodule imports it.

Every building.X import runs this package init first, so the env var lands
before the transitive import tensorflow in the model/training modules. This
kills the cuFFT/cuDNN/NUMA/GPU/TF-TRT spew. Level 3 keeps only fatal C++ logs;
the logger level drops the Python-side AudioResample/autograph warnings.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
