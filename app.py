"""Punto de entrada de la interfaz Streamlit de LabGuard AI."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labguard_ai.ui.app import run


if __name__ == "__main__":
    run()
