"""Funciones de imagen y presentacion independientes de Streamlit."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from labguard_ai.core import AnalysisResult, Detection, PpeType

PPE_LABELS = {
    PpeType.LAB_COAT: "Bata de laboratorio",
    PpeType.PROTECTIVE_GLOVES: "Guantes de proteccion",
    PpeType.SAFETY_GOGGLES: "Gafas de seguridad",
    PpeType.LAB_CAP: "Gorro de laboratorio",
    PpeType.FACE_MASK: "Tapabocas",
}

PPE_SHORT_LABELS = {
    PpeType.LAB_COAT: "Bata",
    PpeType.PROTECTIVE_GLOVES: "Guantes",
    PpeType.SAFETY_GOGGLES: "Gafas",
    PpeType.LAB_CAP: "Gorro",
    PpeType.FACE_MASK: "Tapabocas",
}

PPE_COLORS = {
    PpeType.LAB_COAT: (220, 180, 40),
    PpeType.PROTECTIVE_GLOVES: (70, 210, 120),
    PpeType.SAFETY_GOGGLES: (245, 180, 60),
    PpeType.LAB_CAP: (180, 100, 240),
    PpeType.FACE_MASK: (70, 170, 245),
}


@dataclass(frozen=True, slots=True)
class PpeSummary:
    """Separa resultados observados, posibles ausencias y clases no evaluables."""

    recognized: tuple[PpeType, ...]
    possibly_missing: tuple[PpeType, ...]
    unavailable: tuple[PpeType, ...]


@dataclass(frozen=True, slots=True)
class LabelPlacement:
    """Posicion de una etiqueta calculada para no cubrir otras etiquetas."""

    rectangle: tuple[int, int, int, int] | None
    text_origin: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class FaceMaskAbsenceEvidence:
    """Evidencia facial externa asociada a una persona concreta."""

    person_id: str
    mouth_anchor: tuple[int, int] | None
    face_confidence: float
    absence_confidence: float


@dataclass(frozen=True, slots=True)
class WarningMarker:
    """Advertencia que conserva la persona y una ubicacion facial fiable."""

    person_id: str
    anchor: tuple[int, int]
    confidence: float


def decode_uploaded_image(content: bytes) -> np.ndarray:
    """Decodifica JPG o PNG como una imagen BGR de OpenCV."""
    if not content:
        raise ValueError("El archivo de imagen esta vacio")

    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("El archivo no contiene una imagen JPG o PNG valida")
    return image


def ppe_label(item: PpeType) -> str:
    """Devuelve el nombre en espanol de un EPP."""
    return PPE_LABELS[item]


def detection_label(detection: Detection) -> str:
    """Construye una etiqueta compacta para una deteccion individual."""
    return f"{PPE_SHORT_LABELS[detection.item]} {detection.confidence:.0%}"


def count_detections(detections: Iterable[Detection]) -> Mapping[PpeType, int]:
    """Cuenta instancias individuales sin deduplicarlas por categoria."""
    return dict(Counter(detection.item for detection in detections))


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    padding: int = 2,
) -> bool:
    return not (
        first[2] + padding < second[0]
        or second[2] + padding < first[0]
        or first[3] + padding < second[1]
        or second[3] + padding < first[1]
    )


def layout_detection_labels(
    image_shape: tuple[int, ...],
    detections: Iterable[Detection],
    avoid_regions: Iterable[tuple[int, int, int, int]] = (),
) -> tuple[LabelPlacement, ...]:
    """Distribuye etiquetas cercanas a sus cajas sin superponerlas entre si."""
    height, width = image_shape[:2]
    font_scale = min(0.46, max(0.34, min(height, width) / 1800))
    occupied: list[tuple[int, int, int, int]] = []
    protected = [
        (
            min(max(left, 0), max(width - 1, 0)),
            min(max(top, 0), max(height - 1, 0)),
            min(max(right, 0), max(width - 1, 0)),
            min(max(bottom, 0), max(height - 1, 0)),
        )
        for left, top, right, bottom in avoid_regions
    ]
    placements: list[LabelPlacement] = []

    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        x1 = min(max(x1, 0), max(width - 1, 0))
        x2 = min(max(x2, 0), max(width - 1, 0))
        y1 = min(max(y1, 0), max(height - 1, 0))
        y2 = min(max(y2, 0), max(height - 1, 0))
        label = detection_label(detection)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        label_width = min(width, text_width + 10)
        label_height = min(height, text_height + baseline + 8)
        maximum_x = max(width - label_width, 0)
        maximum_y = max(height - label_height, 0)
        preferred_y = min(max(y1 - label_height - 3, 0), maximum_y)

        x_candidates = [
            min(max(x1, 0), maximum_x),
            min(max(x2 - label_width, 0), maximum_x),
        ]
        y_candidates = [preferred_y, min(y2 + 3, maximum_y)]
        y_candidates.extend(range(0, maximum_y + 1, label_height + 3))
        y_candidates = sorted(set(y_candidates), key=lambda value: abs(value - preferred_y))

        selected: tuple[int, int, int, int] | None = None
        for candidate_y in y_candidates:
            for candidate_x in dict.fromkeys(x_candidates):
                rectangle = (
                    candidate_x,
                    candidate_y,
                    candidate_x + label_width - 1,
                    candidate_y + label_height - 1,
                )
                overlaps_label = any(
                    _rectangles_overlap(rectangle, other) for other in occupied
                )
                overlaps_protected = any(
                    _rectangles_overlap(rectangle, region, padding=0) for region in protected
                )
                if not overlaps_label and not overlaps_protected:
                    selected = rectangle
                    break
            if selected is not None:
                break

        if selected is None:
            placements.append(LabelPlacement(rectangle=None, text_origin=None))
        else:
            occupied.append(selected)
            placements.append(
                LabelPlacement(
                    rectangle=selected,
                    text_origin=(selected[0] + 5, selected[3] - baseline - 3),
                )
            )

    return tuple(placements)


def build_face_mask_warnings(
    evidence: Iterable[FaceMaskAbsenceEvidence],
    *,
    minimum_face_confidence: float = 0.80,
    minimum_absence_confidence: float = 0.80,
) -> tuple[WarningMarker, ...]:
    """Crea advertencias solo con boca localizada y evidencia explicita suficiente."""
    warnings: list[WarningMarker] = []
    for item in evidence:
        if item.mouth_anchor is None:
            continue
        if item.face_confidence < minimum_face_confidence:
            continue
        if item.absence_confidence < minimum_absence_confidence:
            continue
        warnings.append(
            WarningMarker(
                person_id=item.person_id,
                anchor=item.mouth_anchor,
                confidence=item.absence_confidence,
            )
        )
    return tuple(warnings)


def draw_warning_triangle(
    image: np.ndarray,
    anchor: tuple[int, int],
    *,
    size: int = 9,
) -> np.ndarray:
    """Dibuja un pequeno triangulo rojo en una coordenada facial proporcionada."""
    if size < 4:
        raise ValueError("El tamano del icono debe ser al menos 4 pixeles")

    height, width = image.shape[:2]
    center_x = min(max(int(anchor[0]), 0), max(width - 1, 0))
    center_y = min(max(int(anchor[1]), 0), max(height - 1, 0))
    points = np.array(
        [
            (center_x, max(center_y - size, 0)),
            (max(center_x - size, 0), min(center_y + size, height - 1)),
            (min(center_x + size, width - 1), min(center_y + size, height - 1)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], (20, 20, 230))
    cv2.polylines(image, [points], True, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.line(
        image,
        (center_x, max(center_y - size // 2, 0)),
        (center_x, min(center_y + size // 3, height - 1)),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        image,
        (center_x, min(center_y + size // 2, height - 1)),
        1,
        (255, 255, 255),
        -1,
    )
    return image


def annotate_image(
    image: np.ndarray,
    detections: Iterable[Detection],
    *,
    warnings: Iterable[WarningMarker] = (),
    avoid_regions: Iterable[tuple[int, int, int, int]] = (),
) -> np.ndarray:
    """Dibuja cajas y confianza sobre una copia de la imagen BGR."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    box_thickness = max(1, round(min(height, width) / 700))
    font_scale = min(0.46, max(0.34, min(height, width) / 1800))
    detections = tuple(detections)
    warnings = tuple(warnings)

    protected_regions = list(avoid_regions)
    protected_regions.extend(
        detection.bounding_box
        for detection in detections
        if detection.item in {PpeType.SAFETY_GOGGLES, PpeType.FACE_MASK}
    )

    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        x1 = min(max(x1, 0), max(width - 1, 0))
        x2 = min(max(x2, 0), max(width - 1, 0))
        y1 = min(max(y1, 0), max(height - 1, 0))
        y2 = min(max(y2, 0), max(height - 1, 0))
        color = PPE_COLORS[detection.item]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

    placements = layout_detection_labels(
        annotated.shape,
        detections,
        protected_regions,
    )
    for detection, placement in zip(detections, placements, strict=True):
        if placement.rectangle is None or placement.text_origin is None:
            continue
        color = PPE_COLORS[detection.item]
        label = detection_label(detection)
        left, top, right, bottom = placement.rectangle
        cv2.rectangle(annotated, (left, top), (right, bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            placement.text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 23, 31),
            1,
            cv2.LINE_AA,
        )

    for warning in warnings:
        draw_warning_triangle(annotated, warning.anchor)

    return annotated


def summarize_result(
    result: AnalysisResult,
    supported_items: Iterable[PpeType],
) -> PpeSummary:
    """Resume solo las clases que el modelo actual puede evaluar."""
    supported = frozenset(supported_items)
    visible = result.compliance.visible

    return PpeSummary(
        recognized=tuple(item for item in PpeType if item in visible and item in supported),
        possibly_missing=tuple(item for item in PpeType if item in supported - visible),
        unavailable=tuple(item for item in PpeType if item not in supported),
    )
