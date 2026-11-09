"""Reglas basicas de evaluacion de EPP."""

from collections.abc import Iterable

from labguard_ai.core import ComplianceAssessment, Detection, PpeType


class DefaultComplianceEvaluator:
    """Compara el EPP detectado con los elementos requeridos."""

    def __init__(self, required: Iterable[PpeType] | None = None) -> None:
        self.required = frozenset(PpeType if required is None else required)

    def evaluate(self, detections: Iterable[Detection]) -> ComplianceAssessment:
        visible = frozenset(detection.item for detection in detections)
        possibly_missing = self.required - visible

        return ComplianceAssessment(
            visible=visible,
            possibly_missing=possibly_missing,
            is_compliant=not possibly_missing,
        )
