"""Pruebas del contrato, validador y metricas del entrenamiento."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from PIL import Image

from training.evaluate import model_class_names, per_class_metrics
from training.validate_dataset import CLASS_NAMES, SPLITS, validate_dataset


def create_dataset(root: Path) -> Path:
    dataset_root = root / "dataset"
    for index, split in enumerate(SPLITS):
        images = dataset_root / "images" / split
        labels = dataset_root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / f"sample-{index}.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
        annotations = [
            f"{class_id} {0.1 + class_id * 0.16:.2f} 0.5 0.1 0.4"
            for class_id in range(len(CLASS_NAMES))
        ]
        (labels / f"sample-{index}.txt").write_text("\n".join(annotations) + "\n", encoding="utf-8")

    config = root / "dataset.yaml"
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES))
    config.write_text(
        "train: dataset/images/train\n"
        "val: dataset/images/val\n"
        "test: dataset/images/test\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return config


class DatasetValidationTest(unittest.TestCase):
    def test_accepts_complete_dataset_and_counts_classes(self) -> None:
        with TemporaryDirectory() as directory:
            report = validate_dataset(create_dataset(Path(directory)))

        self.assertTrue(report.valid, report.format())
        self.assertEqual(report.image_counts, {"train": 1, "val": 1, "test": 1})
        self.assertEqual(report.annotation_counts, {"train": 6, "val": 6, "test": 6})
        self.assertEqual(report.class_counts["person"], 3)
        self.assertEqual(report.class_counts["protective_gloves"], 3)

    def test_rejects_missing_and_orphan_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_dataset(root)
            (root / "dataset/labels/train/sample-0.txt").unlink()
            (root / "dataset/labels/train/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            report = validate_dataset(config)

        messages = [issue.message for issue in report.issues]
        self.assertIn("falta la etiqueta para sample-0.jpg", messages)
        self.assertIn("etiqueta huerfana sin imagen correspondiente", messages)

    def test_rejects_invalid_class_and_box_outside_image(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_dataset(root)
            (root / "dataset/labels/val/sample-1.txt").write_text(
                "6 0.5 0.5 0.2 0.2\n2 0.95 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            report = validate_dataset(config)

        messages = [issue.message for issue in report.issues]
        self.assertIn("ID de clase 6 fuera del rango 0-5", messages)
        self.assertIn("la caja sale de los limites de la imagen", messages)

    def test_rejects_reordered_class_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_dataset(root)
            text = config.read_text(encoding="utf-8")
            config.write_text(text.replace("0: person", "0: lab_coat"), encoding="utf-8")

            report = validate_dataset(config)

        self.assertIn("names debe declarar exactamente", report.issues[0].message)

    def test_rejects_unsupported_internal_image_format(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_dataset(root)
            image_path = root / "dataset/images/train/sample-0.jpg"
            Image.new("RGB", (20, 20)).save(image_path, format="GIF")

            report = validate_dataset(config)

        self.assertTrue(
            any("formato interno 'gif'" in issue.message for issue in report.issues),
            report.format(),
        )


class FakeBoxMetrics:
    ap_class_index = np.array([0, 2])
    p = np.array([0.8, 0.6])
    r = np.array([0.7, 0.5])
    all_ap = np.array([[0.9] * 10, [0.4] * 10])


class EvaluationMetricsTest(unittest.TestCase):
    def test_associates_metrics_with_present_class_ids(self) -> None:
        rows = per_class_metrics(FakeBoxMetrics())

        self.assertEqual(rows[0]["class_name"], "person")
        self.assertAlmostEqual(rows[0]["map50_95"], 0.9)
        self.assertIsNone(rows[1]["precision"])
        self.assertEqual(rows[2]["class_name"], "protective_gloves")
        self.assertAlmostEqual(rows[2]["map50"], 0.4)

    def test_normalizes_checkpoint_names_only_for_contiguous_ids(self) -> None:
        model = type("Model", (), {"names": dict(enumerate(CLASS_NAMES))})()
        invalid_model = type("Model", (), {"names": {1: "person"}})()

        self.assertEqual(model_class_names(model), CLASS_NAMES)
        self.assertIsNone(model_class_names(invalid_model))


if __name__ == "__main__":
    unittest.main()
