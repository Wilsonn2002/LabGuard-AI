"""Analisis de imagenes para detectar y evaluar EPP."""

from labguard_ai.core import AnalysisResult
from labguard_ai.detection import PpeDetector
from labguard_ai.compliance.rules import DefaultComplianceEvaluator


class ImageAnalyzer:
    def __init__(self, detector: PpeDetector) -> None:
        self.detector = detector
        self.evaluator = DefaultComplianceEvaluator()

    def analyze(self, image: object) -> AnalysisResult:
        detections = tuple(self.detector.detect(image))
        compliance = self.evaluator.evaluate(detections)

        return AnalysisResult(
            detections=detections,
            compliance=compliance,
        )
