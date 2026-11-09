"""Pruebas del evaluador de cumplimiento."""

import unittest

from labguard_ai.core import Detection, PpeType
from labguard_ai.compliance.rules import DefaultComplianceEvaluator


class ComplianceRulesTest(unittest.TestCase):

    def test_detects_possibly_missing_equipment(self):
        evaluator = DefaultComplianceEvaluator()

        detections = [
            Detection(PpeType.LAB_COAT, 0.95, (10, 10, 100, 200)),
            Detection(PpeType.FACE_MASK, 0.90, (20, 20, 80, 80)),
        ]

        result = evaluator.evaluate(detections)

        self.assertFalse(result.is_compliant)
        self.assertEqual(len(result.visible), 2)
        self.assertEqual(len(result.possibly_missing), 3)

    def test_all_equipment_present(self):
        evaluator = DefaultComplianceEvaluator()

        detections = [
            Detection(item, 0.95, (0, 0, 100, 100))
            for item in PpeType
        ]

        result = evaluator.evaluate(detections)

        self.assertTrue(result.is_compliant)
        self.assertEqual(len(result.possibly_missing), 0)


if __name__ == "__main__":
    unittest.main()
