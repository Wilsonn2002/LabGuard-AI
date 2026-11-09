"""Pruebas de la configuracion del modelo sin requerir PyYAML."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labguard_ai.config import ModelConfigurationError, load_model_configuration
from labguard_ai.core import PpeType


def valid_configuration() -> dict:
    return {
        "model": {
            "repository": "owner/model",
            "revision": "abc123",
            "local_path": "models/best.pt",
        },
        "class_mapping": {
            "Gloves": "protective_gloves",
            "goggles": "safety_goggles",
            "mask": "face_mask",
        },
        "person_detection": {
            "local_path": "yolov8n.pt",
            "confidence_threshold": 0.30,
            "iou_threshold": 0.40,
            "maximum_people": 25,
        },
        "inference": {
            "device": "cpu",
            "image_size": 512,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
        },
    }


class ModelConfigurationTest(unittest.TestCase):
    def test_loads_paths_mapping_and_inference_parameters(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "model.yaml"
            with patch("labguard_ai.config._read_yaml", return_value=valid_configuration()):
                configuration = load_model_configuration(config_path, project_root=root)

        self.assertEqual(configuration.model_path, (root / "models" / "best.pt").resolve())
        self.assertEqual(configuration.device, "cpu")
        self.assertEqual(configuration.image_size, 512)
        self.assertEqual(configuration.person_model_path, (root / "yolov8n.pt").resolve())
        self.assertEqual(configuration.person_confidence_threshold, 0.30)
        self.assertEqual(configuration.maximum_people, 25)
        self.assertEqual(
            configuration.class_mapping,
            {
                "Gloves": PpeType.PROTECTIVE_GLOVES,
                "goggles": PpeType.SAFETY_GOGGLES,
                "mask": PpeType.FACE_MASK,
            },
        )

    def test_rejects_non_cpu_device(self) -> None:
        data = valid_configuration()
        data["inference"]["device"] = "cuda:0"

        with patch("labguard_ai.config._read_yaml", return_value=data):
            with self.assertRaises(ModelConfigurationError):
                load_model_configuration("model.yaml")

    def test_rejects_unknown_target_class(self) -> None:
        data = valid_configuration()
        data["class_mapping"]["Vest"] = "lab_coat_from_vest"

        with patch("labguard_ai.config._read_yaml", return_value=data):
            with self.assertRaises(ModelConfigurationError):
                load_model_configuration("model.yaml")


if __name__ == "__main__":
    unittest.main()
