"""Contrato para exportar resultados sin acoplarlos a un formato."""

from pathlib import Path
from typing import Protocol

from labguard_ai.core import AnalysisResult


class ReportGenerator(Protocol):
    """Interfaz para futuros reportes JSON, CSV o PDF."""

    def generate(self, result: AnalysisResult, output_directory: Path) -> Path:
        """Genera un reporte y devuelve su ruta."""
        ...
