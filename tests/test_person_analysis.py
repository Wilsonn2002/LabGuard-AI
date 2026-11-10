"""Pruebas de deteccion global y asociacion de EPP por persona."""

import unittest

import numpy as np

from labguard_ai.core import Detection, EvaluationStatus, PersonDetection, PpeType
from labguard_ai.processing.person import PersonPpeAnalyzer


class FakePersonDetector:
    def __init__(self, people) -> None:
        self.people = people

    def detect(self, image):
        return self.people


class FullImageOnlyPpeDetector:
    def __init__(self, image_shape, detections) -> None:
        self.image_shape = image_shape
        self.detections = detections
        self.seen_shapes = []

    def detect(self, image):
        self.seen_shapes.append(image.shape)
        return self.detections if image.shape == self.image_shape else []


def status_for(analysis, item):
    return next(status.status for status in analysis.statuses if status.item == item)


class PersonPpeAnalyzerTest(unittest.TestCase):
    def test_preserves_global_detections_in_individual_analysis(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detection = Detection(PpeType.LAB_COAT, 0.91, (20, 25, 75, 95))
        ppe_detector = FullImageOnlyPpeDetector(image.shape, [detection])
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector([PersonDetection("Persona 1", 0.95, (10, 5, 90, 100))]),
            ppe_detector,
            {PpeType.LAB_COAT},
        )

        result = analyzer.analyze(image)

        self.assertEqual(ppe_detector.seen_shapes, [image.shape])
        self.assertEqual(result.people[0].detections, (detection,))

    def test_clips_people_and_global_detections_to_the_image(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        people = [
            PersonDetection("Persona 1", 0.95, (-10, 10, 80, 95)),
            PersonDetection("Persona 2", 0.91, (100, 20, 215, 120)),
        ]
        ppe_detector = FullImageOnlyPpeDetector(
            image.shape,
            [
                Detection(PpeType.PROTECTIVE_GLOVES, 0.88, (-5, 15, 20, 35)),
                Detection(PpeType.SAFETY_GOGGLES, 0.82, (110, 30, 130, 45)),
            ],
        )
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector(people),
            ppe_detector,
            {
                PpeType.PROTECTIVE_GLOVES,
                PpeType.SAFETY_GOGGLES,
                PpeType.FACE_MASK,
            },
        )

        result = analyzer.analyze(image)

        self.assertEqual(ppe_detector.seen_shapes, [image.shape])
        self.assertEqual(result.people[0].person.bounding_box, (0, 10, 80, 95))
        self.assertEqual(result.people[1].person.bounding_box, (100, 20, 200, 100))
        self.assertEqual(result.people[0].detections[0].bounding_box, (0, 15, 20, 35))
        self.assertEqual(result.people[1].detections[0].bounding_box, (110, 30, 130, 45))
        self.assertEqual(
            status_for(result.people[0], PpeType.PROTECTIVE_GLOVES),
            EvaluationStatus.DETECTED,
        )
        self.assertEqual(
            status_for(result.people[0], PpeType.SAFETY_GOGGLES),
            EvaluationStatus.UNDETERMINED,
        )
        self.assertEqual(
            status_for(result.people[0], PpeType.LAB_COAT),
            EvaluationStatus.NOT_EVALUABLE,
        )

    def test_assigns_an_object_to_only_one_overlapping_person(self) -> None:
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        people = [
            PersonDetection("Persona 1", 0.95, (0, 0, 100, 100)),
            PersonDetection("Persona 2", 0.94, (50, 0, 150, 100)),
        ]
        ppe_detector = FullImageOnlyPpeDetector(
            image.shape,
            [Detection(PpeType.PROTECTIVE_GLOVES, 0.91, (80, 20, 100, 40))],
        )
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector(people),
            ppe_detector,
            {PpeType.PROTECTIVE_GLOVES},
        )

        result = analyzer.analyze(image)

        self.assertFalse(result.people[0].detections)
        self.assertEqual(len(result.people[1].detections), 1)
        self.assertEqual(
            result.people[1].detections[0].bounding_box,
            (80, 20, 100, 40),
        )

    def test_distinct_objects_remain_separate_between_people(self) -> None:
        image = np.zeros((100, 170, 3), dtype=np.uint8)
        people = [
            PersonDetection("Persona 1", 0.95, (0, 0, 80, 100)),
            PersonDetection("Persona 2", 0.94, (70, 0, 160, 100)),
        ]
        ppe_detector = FullImageOnlyPpeDetector(
            image.shape,
            [
                Detection(PpeType.SAFETY_GOGGLES, 0.90, (10, 10, 30, 25)),
                Detection(PpeType.SAFETY_GOGGLES, 0.87, (115, 10, 135, 25)),
            ],
        )
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector(people),
            ppe_detector,
            {PpeType.SAFETY_GOGGLES},
        )

        result = analyzer.analyze(image)

        self.assertEqual(len(result.people[0].detections), 1)
        self.assertEqual(len(result.people[1].detections), 1)
        self.assertNotEqual(
            result.people[0].detections[0].bounding_box,
            result.people[1].detections[0].bounding_box,
        )

    def test_keeps_non_overlapping_detections_unassigned(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detection = Detection(PpeType.FACE_MASK, 0.84, (150, 10, 180, 35))
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector([PersonDetection("Persona 1", 0.95, (10, 5, 90, 100))]),
            FullImageOnlyPpeDetector(image.shape, [detection]),
            {PpeType.FACE_MASK},
        )

        result = analyzer.analyze(image)

        self.assertFalse(result.people[0].detections)
        self.assertEqual(result.unassigned_detections, (detection,))


if __name__ == "__main__":
    unittest.main()
