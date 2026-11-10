"""Pruebas de decodificacion, anotacion y resumen para la interfaz."""

import unittest

import cv2
import numpy as np

from labguard_ai.core import (
    AnalysisResult,
    ComplianceAssessment,
    Detection,
    PersonAnalysis,
    PersonDetection,
    PpeType,
)
from labguard_ai.ui.images import (
    FaceMaskAbsenceEvidence,
    annotate_image,
    annotate_people,
    build_face_mask_warnings,
    count_detections,
    decode_uploaded_image,
    detection_label,
    draw_warning_triangle,
    layout_detection_labels,
    summarize_result,
)


class UiImagesTest(unittest.TestCase):
    def test_decodes_png_bytes_as_opencv_image(self) -> None:
        source = np.zeros((20, 30, 3), dtype=np.uint8)
        source[:, :, 1] = 180
        encoded_ok, encoded = cv2.imencode(".png", source)
        self.assertTrue(encoded_ok)

        decoded = decode_uploaded_image(encoded.tobytes())

        np.testing.assert_array_equal(decoded, source)

    def test_rejects_invalid_image_content(self) -> None:
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"not an image")

    def test_draws_detection_without_modifying_original(self) -> None:
        original = np.zeros((100, 120, 3), dtype=np.uint8)
        detection = Detection(PpeType.FACE_MASK, 0.91, (10, 20, 80, 90))

        annotated = annotate_image(original, [detection])

        self.assertFalse(np.array_equal(annotated, original))
        self.assertFalse(original.any())

    def test_draws_person_with_assigned_ppe_count(self) -> None:
        original = np.zeros((100, 120, 3), dtype=np.uint8)
        analysis = PersonAnalysis(
            person=PersonDetection("Persona 1", 0.95, (10, 10, 100, 95)),
            detections=(Detection(PpeType.FACE_MASK, 0.91, (35, 20, 65, 45)),),
            statuses=(),
        )

        annotated = annotate_people(original, [analysis])

        self.assertFalse(np.array_equal(annotated, original))
        self.assertFalse(original.any())

    def test_uses_compact_detection_labels(self) -> None:
        self.assertEqual(
            detection_label(Detection(PpeType.PROTECTIVE_GLOVES, 0.90, (0, 0, 1, 1))),
            "Guantes 90%",
        )
        self.assertEqual(
            detection_label(Detection(PpeType.SAFETY_GOGGLES, 0.85, (0, 0, 1, 1))),
            "Gafas 85%",
        )
        self.assertEqual(
            detection_label(Detection(PpeType.FACE_MASK, 0.91, (0, 0, 1, 1))),
            "Tapabocas 91%",
        )

    def test_layout_avoids_overlapping_labels_for_nearby_detections(self) -> None:
        detections = [
            Detection(PpeType.PROTECTIVE_GLOVES, 0.91, (40, 60, 150, 180)),
            Detection(PpeType.SAFETY_GOGGLES, 0.84, (42, 62, 152, 182)),
            Detection(PpeType.FACE_MASK, 0.79, (44, 64, 154, 184)),
        ]

        placements = layout_detection_labels((240, 480, 3), detections)

        self.assertEqual(len(placements), 3)
        for index, placement in enumerate(placements):
            left, top, right, bottom = placement.rectangle
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLess(right, 480)
            self.assertLess(bottom, 240)
            for other in placements[index + 1 :]:
                other_left, other_top, other_right, other_bottom = other.rectangle
                overlaps = not (
                    right < other_left
                    or other_right < left
                    or bottom < other_top
                    or other_bottom < top
                )
                self.assertFalse(overlaps)

    def test_layout_respects_explicit_face_regions(self) -> None:
        detection = Detection(PpeType.PROTECTIVE_GLOVES, 0.91, (30, 35, 100, 90))
        face_region = (0, 0, 180, 105)

        placement = layout_detection_labels(
            (220, 320, 3),
            [detection],
            avoid_regions=[face_region],
        )[0]

        self.assertIsNotNone(placement.rectangle)
        self.assertGreater(placement.rectangle[1], face_region[3])

    def test_counts_each_detection_instance_by_category(self) -> None:
        detections = [
            Detection(PpeType.PROTECTIVE_GLOVES, 0.91, (0, 0, 10, 10)),
            Detection(PpeType.PROTECTIVE_GLOVES, 0.83, (20, 0, 30, 10)),
            Detection(PpeType.FACE_MASK, 0.88, (40, 0, 50, 10)),
        ]

        counts = count_detections(detections)

        self.assertEqual(counts[PpeType.PROTECTIVE_GLOVES], 2)
        self.assertEqual(counts[PpeType.FACE_MASK], 1)

    def test_draws_small_red_warning_triangle_at_provided_anchor(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)

        returned = draw_warning_triangle(image, (40, 40), size=8)

        self.assertIs(returned, image)
        self.assertGreater(np.count_nonzero(image[:, :, 2] > image[:, :, 1]), 0)

    def test_warning_requires_face_location_and_explicit_absence_evidence(self) -> None:
        evidence = [
            FaceMaskAbsenceEvidence("person-1", (20, 30), 0.94, 0.91),
            FaceMaskAbsenceEvidence("person-2", None, 0.96, 0.93),
            FaceMaskAbsenceEvidence("person-3", (70, 30), 0.75, 0.95),
            FaceMaskAbsenceEvidence("person-4", (90, 30), 0.96, 0.55),
        ]

        warnings = build_face_mask_warnings(evidence)

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].person_id, "person-1")
        self.assertEqual(warnings[0].anchor, (20, 30))

    def test_warnings_remain_separate_for_different_people(self) -> None:
        warnings = build_face_mask_warnings(
            [
                FaceMaskAbsenceEvidence("person-1", (20, 30), 0.95, 0.90),
                FaceMaskAbsenceEvidence("person-2", (80, 35), 0.92, 0.88),
            ]
        )

        self.assertEqual([warning.person_id for warning in warnings], ["person-1", "person-2"])
        self.assertEqual([warning.anchor for warning in warnings], [(20, 30), (80, 35)])

    def test_summary_does_not_mark_unsupported_classes_as_missing(self) -> None:
        assessment = ComplianceAssessment(
            visible=frozenset({PpeType.FACE_MASK}),
            possibly_missing=frozenset(PpeType) - {PpeType.FACE_MASK},
            is_compliant=False,
        )
        result = AnalysisResult(
            detections=(Detection(PpeType.FACE_MASK, 0.9, (0, 0, 10, 10)),),
            compliance=assessment,
        )

        summary = summarize_result(
            result,
            {PpeType.PROTECTIVE_GLOVES, PpeType.SAFETY_GOGGLES, PpeType.FACE_MASK},
        )

        self.assertEqual(summary.recognized, (PpeType.FACE_MASK,))
        self.assertEqual(
            summary.possibly_missing,
            (PpeType.PROTECTIVE_GLOVES, PpeType.SAFETY_GOGGLES),
        )
        self.assertEqual(summary.unavailable, (PpeType.LAB_COAT, PpeType.LAB_CAP))


if __name__ == "__main__":
    unittest.main()
