"""Modelos de datos independientes de YOLO, OpenCV y Streamlit."""

from dataclasses import dataclass, field
from enum import StrEnum


class PpeType(StrEnum):
    """Elementos de proteccion personal evaluados por la aplicacion."""

    LAB_COAT = "lab_coat"
    PROTECTIVE_GLOVES = "protective_gloves"
    SAFETY_GOGGLES = "safety_goggles"
    LAB_CAP = "lab_cap"
    FACE_MASK = "face_mask"


class EvaluationStatus(StrEnum):
    """Estado prudente de un EPP para una persona detectada."""

    DETECTED = "detected"
    UNDETERMINED = "undetermined"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True, slots=True)
class Detection:
    """Deteccion individual producida por el modelo."""

    item: PpeType
    confidence: float
    bounding_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """Persona identificada temporalmente dentro de una fotografia."""

    person_id: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PersonPpeStatus:
    """Estado de un tipo de EPP para una persona concreta."""

    item: PpeType
    status: EvaluationStatus


@dataclass(frozen=True, slots=True)
class PersonAnalysis:
    """Detecciones y estados asociados a una sola persona."""

    person: PersonDetection
    detections: tuple[Detection, ...]
    statuses: tuple[PersonPpeStatus, ...]


@dataclass(frozen=True, slots=True)
class PersonAnalysisResult:
    """Resultado por persona y detecciones EPP que no pudieron asociarse."""

    people: tuple[PersonAnalysis, ...]
    unassigned_detections: tuple[Detection, ...] = ()


@dataclass(frozen=True, slots=True)
class ComplianceAssessment:
    """Elementos visibles y posiblemente faltantes en un analisis."""

    visible: frozenset[PpeType] = field(default_factory=frozenset)
    possibly_missing: frozenset[PpeType] = field(default_factory=frozenset)
    is_compliant: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Resultado comun para imagenes, fotogramas y reportes."""

    detections: tuple[Detection, ...]
    compliance: ComplianceAssessment
