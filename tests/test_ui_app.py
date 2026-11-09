"""Pruebas de construccion del analizador con ajustes de interfaz."""

from pathlib import Path
import unittest
from unittest.mock import patch

from labguard_ai.config import ModelConfiguration
from labguard_ai.core import PpeType
from labguard_ai.ui.app import _analyzer_key, build_analyzer


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


if __name__ == "__main__":
    unittest.main()
