"""Pruebas del detector COCO de personas sin cargar pesos reales."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from labguard_ai.detection import YoloPersonDetector


class FakeBoxes:
    def __init__(self, confidences, coordinates) -> None:
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.xyxy = np.asarray(coordinates, dtype=np.float32)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeModel:
    def __init__(self, results) -> None:
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class YoloPersonDetectorTest(unittest.TestCase):
    def test_uses_only_coco_person_and_assigns_left_to_right_ids(self) -> None:
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolov8n.pt"
            model_path.write_bytes(b"simulated model")
            model = FakeModel(
                [
                    FakeResult(
                        FakeBoxes(
                            [0.88, 0.94],
                            [[120, 10, 190, 100], [10, 5, 80, 110]],
                        )
                    )
                ]
            )
            detector = YoloPersonDetector(
                model_path,
                device="cpu",
                confidence_threshold=0.25,
                image_size=640,
                model_loader=lambda _: model,
            )

            people = detector.detect(np.zeros((120, 220, 3), dtype=np.uint8))

        self.assertEqual([person.person_id for person in people], ["Persona 1", "Persona 2"])
        self.assertEqual(people[0].bounding_box, (10, 5, 80, 110))
        self.assertEqual(model.calls[0]["classes"], [0])
        self.assertEqual(model.calls[0]["device"], "cpu")
        self.assertEqual(model.calls[0]["imgsz"], 640)


if __name__ == "__main__":
    unittest.main()
