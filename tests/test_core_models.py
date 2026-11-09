"""Pruebas basicas de los tipos compartidos del proyecto."""

import unittest

from labguard_ai.core import Detection, PpeType


class CoreModelsTest(unittest.TestCase):
    def test_all_required_ppe_types_are_defined(self) -> None:
        self.assertEqual(
            {item.value for item in PpeType},
            {"lab_coat", "protective_gloves", "safety_goggles", "lab_cap", "face_mask"},
        )

    def test_detection_keeps_confidence_and_box(self) -> None:
        detection = Detection(PpeType.FACE_MASK, 0.91, (10, 20, 30, 40))

        self.assertEqual(detection.confidence, 0.91)
        self.assertEqual(detection.bounding_box, (10, 20, 30, 40))


if __name__ == "__main__":
    unittest.main()
