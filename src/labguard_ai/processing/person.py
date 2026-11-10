"""Analisis de EPP independiente para cada persona de una fotografia."""

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


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _association_score(
    detection: Detection,
    person: PersonDetection,
) -> tuple[float, float] | None:
    intersection = _intersection_area(detection.bounding_box, person.bounding_box)
    detection_area = _area(detection.bounding_box)
    if intersection == 0 or detection_area == 0:
        return None

    detection_x = (detection.bounding_box[0] + detection.bounding_box[2]) / 2
    detection_y = (detection.bounding_box[1] + detection.bounding_box[3]) / 2
    person_x = (person.bounding_box[0] + person.bounding_box[2]) / 2
    person_y = (person.bounding_box[1] + person.bounding_box[3]) / 2
    center_distance_squared = (
        (detection_x - person_x) ** 2 + (detection_y - person_y) ** 2
    )
    return intersection / detection_area, -center_distance_squared


class PersonPpeAnalyzer:
    """Detecta sobre la imagen completa y asocia cada EPP con una persona."""

    def __init__(
        self,
        person_detector: PersonDetector,
        ppe_detector: PpeDetector,
        supported_items: set[PpeType] | frozenset[PpeType],
    ) -> None:
        self.person_detector = person_detector
        self.ppe_detector = ppe_detector
        self.supported_items = frozenset(supported_items)

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
    def _clip_detection(
        detection: Detection,
        image_width: int,
        image_height: int,
    ) -> Detection | None:
        x1, y1, x2, y2 = detection.bounding_box
        clipped = (
            min(max(x1, 0), image_width),
            min(max(y1, 0), image_height),
            min(max(x2, 0), image_width),
            min(max(y2, 0), image_height),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return Detection(detection.item, detection.confidence, clipped)

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
        detections_by_person: list[list[Detection]] = [[] for _ in people]
        unassigned_detections: list[Detection] = []
        detections = (
            clipped
            for detection in self.ppe_detector.detect(image)
            if (
                clipped := self._clip_detection(
                    detection,
                    image_width,
                    image_height,
                )
            )
            is not None
        )
        for detection in detections:
            scored_people = (
                (score, person_index)
                for person_index, person in enumerate(people)
                if (score := _association_score(detection, person)) is not None
            )
            best_match = max(scored_people, default=None)
            if best_match is None:
                unassigned_detections.append(detection)
                continue
            detections_by_person[best_match[1]].append(detection)

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

        return PersonAnalysisResult(
            people=tuple(analyses),
            unassigned_detections=tuple(unassigned_detections),
        )
