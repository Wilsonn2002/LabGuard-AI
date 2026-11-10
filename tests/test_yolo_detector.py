"""Pruebas del adaptador YOLOv8 sin descargar ni cargar pesos reales."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from labguard_ai.core import PpeType
from labguard_ai.detection import YoloPpeDetector
from labguard_ai.processing.image import ImageAnalyzer


class FakeBoxes:
    def __init__(self, classes, confidences, coordinates) -> None:
        self.cls = np.asarray(classes, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.xyxy = np.asarray(coordinates, dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: FakeBoxes, names) -> None:
        self.boxes = boxes
        self.names = names


class FakeYoloModel:
    def __init__(self, results) -> None:
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class YoloPpeDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.model_path = Path(self.temporary_directory.name) / "specialized-ppe.pt"
        self.model_path.write_bytes(b"simulated model")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_converts_mapped_predictions_and_ignores_unknown_classes(self) -> None:
        model = FakeYoloModel(
            [
                FakeResult(
                    FakeBoxes(
                        classes=[0, 1, 2],
                        confidences=[0.91, 0.82, 0.99],
                        coordinates=[
                            [1.2, 2.6, 9.7, 20.4],
                            [5.0, 6.0, 15.0, 16.0],
                            [0.0, 0.0, 25.0, 25.0],
                        ],
                    ),
                    names={0: "coat", 1: "gloves", 2: "person"},
                )
            ]
        )
        loaded_paths = []
        detector = YoloPpeDetector(
            self.model_path,
            {"coat": PpeType.LAB_COAT, "gloves": "protective_gloves"},
            model_loader=lambda path: loaded_paths.append(path) or model,
        )

        detections = detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))

        self.assertEqual([item.item for item in detections], [PpeType.LAB_COAT, PpeType.PROTECTIVE_GLOVES])
        self.assertEqual(detections[0].bounding_box, (1, 3, 10, 20))
        self.assertAlmostEqual(detections[0].confidence, 0.91, places=5)
        self.assertEqual(loaded_paths, [str(self.model_path)])
        self.assertEqual(model.calls[0]["device"], "cpu")
        self.assertEqual(model.calls[0]["imgsz"], 512)
        self.assertIs(model.calls[0]["source"].dtype.type, np.uint8)

    def test_accepts_id_mapping_and_converts_grayscale_with_opencv(self) -> None:
        model = FakeYoloModel(
            [
                FakeResult(
                    FakeBoxes([7], [0.88], [[2, 3, 8, 9]]),
                    names={7: "custom-mask-label"},
                )
            ]
        )
        detector = YoloPpeDetector(
            self.model_path,
            {7: PpeType.FACE_MASK},
            model_loader=lambda _: model,
        )

        detections = detector.detect(np.zeros((12, 16), dtype=np.uint8))

        self.assertEqual(detections[0].item, PpeType.FACE_MASK)
        self.assertEqual(model.calls[0]["source"].shape, (12, 16, 3))

    def test_maps_all_definitive_ppe_classes_and_ignores_person(self) -> None:
        names = {
            0: "person",
            1: "lab_coat",
            2: "protective_gloves",
            3: "safety_goggles",
            4: "lab_cap",
            5: "face_mask",
        }
        model = FakeYoloModel(
            [
                FakeResult(
                    FakeBoxes(
                        classes=list(names),
                        confidences=[0.90] * len(names),
                        coordinates=[[0, 0, 10, 10]] * len(names),
                    ),
                    names=names,
                )
            ]
        )
        mapping = {item.value: item for item in PpeType}
        detector = YoloPpeDetector(
            self.model_path,
            mapping,
            model_loader=lambda _: model,
        )

        detections = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual([detection.item for detection in detections], list(PpeType))

    def test_integrates_with_image_analyzer_and_compliance_evaluator(self) -> None:
        class_names = {index: item.value for index, item in enumerate(PpeType)}
        model = FakeYoloModel(
            [
                FakeResult(
                    FakeBoxes(
                        classes=list(class_names),
                        confidences=[0.9] * len(class_names),
                        coordinates=[[0, 0, 10, 10]] * len(class_names),
                    ),
                    names=class_names,
                )
            ]
        )
        detector = YoloPpeDetector(self.model_path, model_loader=lambda _: model)

        result = ImageAnalyzer(detector).analyze(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(len(result.detections), len(PpeType))
        self.assertTrue(result.compliance.is_compliant)
        self.assertFalse(result.compliance.possibly_missing)

    def test_rejects_missing_model_to_avoid_implicit_downloads(self) -> None:
        missing_model = Path(self.temporary_directory.name) / "missing.pt"

        with self.assertRaises(FileNotFoundError):
            YoloPpeDetector(missing_model)

    def test_rejects_non_numpy_images(self) -> None:
        detector = YoloPpeDetector(
            self.model_path,
            model_loader=lambda _: FakeYoloModel([]),
        )

        with self.assertRaises(TypeError):
            detector.detect("image.jpg")


if __name__ == "__main__":
    unittest.main()
