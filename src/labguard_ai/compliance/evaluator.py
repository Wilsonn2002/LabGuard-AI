"""Contrato para determinar EPP visible y posiblemente faltante."""

from collections.abc import Iterable
from typing import Protocol

from labguard_ai.core import ComplianceAssessment, Detection


class ComplianceEvaluator(Protocol):
    """Evalua detecciones sin depender de la interfaz ni del modelo usado."""

    def evaluate(self, detections: Iterable[Detection]) -> ComplianceAssessment:
        """Convierte detecciones en una evaluacion de cumplimiento."""
        ...
