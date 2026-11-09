"""Pruebas del analisis EPP por persona, recortes y deduplicacion."""

import unittest

import numpy as np

from labguard_ai.core import Detection, EvaluationStatus, PersonDetection, PpeType
from labguard_ai.processing.person import PersonPpeAnalyzer


class FakePersonDetector:
    def __init__(self, people) -> None:
        self.people = people

    def detect(self, image):
        return self.people


class SequentialPpeDetector:
    def __init__(self, outputs) -> None:
        self.outputs = iter(outputs)
        self.crop_shapes = []

    def detect(self, image):
        self.crop_shapes.append(image.shape)
        return next(self.outputs)


def status_for(analysis, item):
    return next(status.status for status in analysis.statuses if status.item == item)


class PersonPpeAnalyzerTest(unittest.TestCase):
    def test_clips_people_crops_and_translates_partial_detections(self) -> None:
        people = [
            PersonDetection("Persona 1", 0.95, (-10, 10, 80, 95)),
            PersonDetection("Persona 2", 0.91, (100, 20, 215, 120)),
        ]
        ppe_detector = SequentialPpeDetector(
            [
                [Detection(PpeType.PROTECTIVE_GLOVES, 0.88, (-5, 5, 20, 25))],
                [Detection(PpeType.SAFETY_GOGGLES, 0.82, (10, 10, 30, 25))],
            ]
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

        result = analyzer.analyze(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(ppe_detector.crop_shapes, [(85, 80, 3), (80, 100, 3)])
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

    def test_overlapping_crops_do_not_assign_same_object_twice(self) -> None:
        people = [
            PersonDetection("Persona 1", 0.95, (0, 0, 100, 100)),
            PersonDetection("Persona 2", 0.94, (50, 0, 150, 100)),
        ]
        ppe_detector = SequentialPpeDetector(
            [
                [Detection(PpeType.PROTECTIVE_GLOVES, 0.80, (60, 20, 80, 40))],
                [Detection(PpeType.PROTECTIVE_GLOVES, 0.91, (10, 20, 30, 40))],
            ]
        )
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector(people),
            ppe_detector,
            {PpeType.PROTECTIVE_GLOVES},
        )

        result = analyzer.analyze(np.zeros((100, 160, 3), dtype=np.uint8))

        self.assertFalse(result.people[0].detections)
        self.assertEqual(len(result.people[1].detections), 1)
        self.assertEqual(
            result.people[1].detections[0].bounding_box,
            (60, 20, 80, 40),
        )

    def test_distinct_objects_remain_separate_between_people(self) -> None:
        people = [
            PersonDetection("Persona 1", 0.95, (0, 0, 80, 100)),
            PersonDetection("Persona 2", 0.94, (70, 0, 160, 100)),
        ]
        ppe_detector = SequentialPpeDetector(
            [
                [Detection(PpeType.SAFETY_GOGGLES, 0.90, (10, 10, 30, 25))],
                [Detection(PpeType.SAFETY_GOGGLES, 0.87, (45, 10, 65, 25))],
            ]
        )
        analyzer = PersonPpeAnalyzer(
            FakePersonDetector(people),
            ppe_detector,
            {PpeType.SAFETY_GOGGLES},
        )

        result = analyzer.analyze(np.zeros((100, 170, 3), dtype=np.uint8))

        self.assertEqual(len(result.people[0].detections), 1)
        self.assertEqual(len(result.people[1].detections), 1)
        self.assertNotEqual(
            result.people[0].detections[0].bounding_box,
            result.people[1].detections[0].bounding_box,
        )


if __name__ == "__main__":
    unittest.main()
