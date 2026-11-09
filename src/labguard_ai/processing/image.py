"""Preparacion y analisis futuro de imagenes cargadas por el usuario."""

from labguard_ai.core import AnalysisResult
from labguard_ai.detection import PpeDetector


class ImageAnalyzer:
    """Coordina la deteccion y evaluacion de una imagen estatica."""

    def __init__(self, detector: PpeDetector) -> None:
        self.detector = detector

    def analyze(self, image: object) -> AnalysisResult:
        """Analiza una imagen cuando se conecte el evaluador de cumplimiento."""
        raise NotImplementedError("El flujo de analisis se implementara en una fase posterior")
