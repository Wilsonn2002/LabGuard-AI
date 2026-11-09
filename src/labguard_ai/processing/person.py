"""Analisis de EPP independiente para cada persona de una fotografia."""

from dataclasses import dataclass

import numpy as np

from labguard_ai.core import (
    Detection,
    EvaluationStatus,
    PersonAnalysis,
    PersonAnalysisResult,
    PersonDetection,
    PersonPpeStatus,
    PpeType,
)
from labguard_ai.detection import PersonDetector, PpeDetector


@dataclass(frozen=True, slots=True)
class _Candidate:
    person_index: int
    detection: Detection


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _same_object(first: Detection, second: Detection, iou_threshold: float) -> bool:
    if first.item != second.item:
        return False
    intersection = _intersection_area(first.bounding_box, second.bounding_box)
    if intersection == 0:
        return False
    first_area = _area(first.bounding_box)
    second_area = _area(second.bounding_box)
    union = first_area + second_area - intersection
    iou = intersection / union if union else 0.0
    smaller_area = min(first_area, second_area)
    overlap_of_smaller = intersection / smaller_area if smaller_area else 0.0
    return iou >= iou_threshold or overlap_of_smaller >= 0.80


class PersonPpeAnalyzer:
    """Recorta cada persona, detecta EPP y resuelve duplicados entre recortes."""

    def __init__(
        self,
        person_detector: PersonDetector,
        ppe_detector: PpeDetector,
        supported_items: set[PpeType] | frozenset[PpeType],
        *,
        duplicate_iou_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold debe estar entre 0 y 1")
        self.person_detector = person_detector
        self.ppe_detector = ppe_detector
        self.supported_items = frozenset(supported_items)
        self.duplicate_iou_threshold = duplicate_iou_threshold

    @staticmethod
    def _clip_person(
        person: PersonDetection,
        image_width: int,
        image_height: int,
    ) -> PersonDetection | None:
        x1, y1, x2, y2 = person.bounding_box
        x1 = min(max(x1, 0), image_width)
        y1 = min(max(y1, 0), image_height)
        x2 = min(max(x2, 0), image_width)
        y2 = min(max(y2, 0), image_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return PersonDetection(person.person_id, person.confidence, (x1, y1, x2, y2))

    @staticmethod
    def _translate_detection(
        detection: Detection,
        offset_x: int,
        offset_y: int,
        image_width: int,
        image_height: int,
    ) -> Detection | None:
        x1, y1, x2, y2 = detection.bounding_box
        translated = (
            min(max(x1 + offset_x, 0), image_width),
            min(max(y1 + offset_y, 0), image_height),
            min(max(x2 + offset_x, 0), image_width),
            min(max(y2 + offset_y, 0), image_height),
        )
        if translated[2] <= translated[0] or translated[3] <= translated[1]:
            return None
        return Detection(detection.item, detection.confidence, translated)

    def analyze(self, image: object) -> PersonAnalysisResult:
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser una matriz NumPy")
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("La imagen no es valida")

        image_height, image_width = image.shape[:2]
        people = tuple(
            clipped
            for person in self.person_detector.detect(image)
            if (clipped := self._clip_person(person, image_width, image_height)) is not None
        )
        candidates: list[_Candidate] = []

        for person_index, person in enumerate(people):
            x1, y1, x2, y2 = person.bounding_box
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            for local_detection in self.ppe_detector.detect(crop):
                translated = self._translate_detection(
                    local_detection,
                    x1,
                    y1,
                    image_width,
                    image_height,
                )
                if translated is not None:
                    candidates.append(_Candidate(person_index, translated))

        accepted: list[_Candidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: item.detection.confidence,
            reverse=True,
        ):
            if any(
                _same_object(
                    candidate.detection,
                    existing.detection,
                    self.duplicate_iou_threshold,
                )
                for existing in accepted
            ):
                continue
            accepted.append(candidate)

        detections_by_person: list[list[Detection]] = [[] for _ in people]
        for candidate in accepted:
            detections_by_person[candidate.person_index].append(candidate.detection)

        analyses: list[PersonAnalysis] = []
        for person, detections in zip(people, detections_by_person, strict=True):
            visible = {detection.item for detection in detections}
            statuses = tuple(
                PersonPpeStatus(
                    item=item,
                    status=(
                        EvaluationStatus.NOT_EVALUABLE
                        if item not in self.supported_items
                        else EvaluationStatus.DETECTED
                        if item in visible
                        else EvaluationStatus.UNDETERMINED
                    ),
                )
                for item in PpeType
            )
            analyses.append(
                PersonAnalysis(
                    person=person,
                    detections=tuple(detections),
                    statuses=statuses,
                )
            )

        return PersonAnalysisResult(people=tuple(analyses))
