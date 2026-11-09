"""Captura y procesamiento futuro de fotogramas de video."""

from collections.abc import Iterable, Iterator

from labguard_ai.core import AnalysisResult


class VideoAnalyzer:
    """Define el flujo incremental usado por camaras y archivos de video."""

    def analyze(self, frames: Iterable[object]) -> Iterator[AnalysisResult]:
        """Produce resultados por fotograma sin retener todo el video en memoria."""
        raise NotImplementedError("El procesamiento de video se implementara en una fase posterior")
