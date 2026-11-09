"""Pruebas de construccion del analizador con ajustes de interfaz."""

from pathlib import Path
import unittest
from unittest.mock import patch

from labguard_ai.config import ModelConfiguration
from labguard_ai.core import PpeType
from labguard_ai.ui.app import (
    _analyzer_key,
    _person_analyzer_key,
    build_analyzer,
    build_person_analyzer,
)


class UiAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = ModelConfiguration(
            model_path=Path("models/best.pt"),
            repository="owner/model",
            revision="abc123",
            class_mapping={"goggles": PpeType.SAFETY_GOGGLES},
            device="cpu",
            image_size=512,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

    @patch("labguard_ai.ui.app.YoloPpeDetector")
    def test_runtime_controls_are_forwarded_to_detector(self, detector_class) -> None:
        analyzer = build_analyzer(
            self.configuration,
            confidence_threshold=0.10,
            image_size=640,
        )

        detector_class.assert_called_once_with(
            self.configuration.model_path,
            self.configuration.class_mapping,
            device="cpu",
            confidence_threshold=0.10,
            iou_threshold=0.45,
            image_size=640,
        )
        self.assertIs(analyzer.detector, detector_class.return_value)

    def test_analyzer_key_changes_with_runtime_parameters(self) -> None:
        baseline = _analyzer_key(self.configuration, 0.25, 512)

        self.assertNotEqual(baseline, _analyzer_key(self.configuration, 0.10, 512))
        self.assertNotEqual(baseline, _analyzer_key(self.configuration, 0.25, 640))

    @patch("labguard_ai.ui.app.PersonPpeAnalyzer")
    @patch("labguard_ai.ui.app.YoloPpeDetector")
    @patch("labguard_ai.ui.app.YoloPersonDetector")
    def test_builds_independent_person_and_ppe_detectors(
        self,
        person_detector_class,
        ppe_detector_class,
        analyzer_class,
    ) -> None:
        self.configuration = ModelConfiguration(
            **{
                **{
                    field: getattr(self.configuration, field)
                    for field in (
                        "model_path",
                        "repository",
                        "revision",
                        "class_mapping",
                        "device",
                        "image_size",
                        "confidence_threshold",
                        "iou_threshold",
                    )
                },
                "person_model_path": Path("yolov8n.pt"),
            }
        )

        build_person_analyzer(
            self.configuration,
            confidence_threshold=0.15,
            image_size=640,
        )

        person_detector_class.assert_called_once_with(
            Path("yolov8n.pt"),
            device="cpu",
            confidence_threshold=0.25,
            iou_threshold=0.45,
            image_size=640,
            maximum_detections=50,
        )
        ppe_detector_class.assert_called_once_with(
            self.configuration.model_path,
            self.configuration.class_mapping,
            device="cpu",
            confidence_threshold=0.15,
            iou_threshold=0.45,
            image_size=640,
        )
        analyzer_class.assert_called_once()

    def test_person_analyzer_key_changes_with_runtime_parameters(self) -> None:
        baseline = _person_analyzer_key(self.configuration, 0.25, 512)

        self.assertNotEqual(
            baseline,
            _person_analyzer_key(self.configuration, 0.15, 512),
        )
        self.assertNotEqual(
            baseline,
            _person_analyzer_key(self.configuration, 0.25, 640),
        )


if __name__ == "__main__":
    unittest.main()
