"""Paths can be redirected for isolated verification and full-pipeline staging."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MACRO_DATA_DIR", ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(os.environ.get("MACRO_FIGURES_DIR", ROOT / "reports" / "figures")).resolve()
