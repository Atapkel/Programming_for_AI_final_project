"""Compatibility launcher for the unified Streamlit application.

Both ``streamlit run app.py`` and ``streamlit run app/main.py`` use the same UI.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "main.py"), run_name="__main__")
