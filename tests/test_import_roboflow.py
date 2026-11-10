"""Pruebas del importador auditable de datasets Roboflow."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from training.import_roboflow import (
    CLASS_MAPPING,
    DatasetImportError,
    SOURCE_CLASS_NAMES,
    import_roboflow_archive,
    parse_annotation_line,
)


DATA_YAML = """train: ../train/images
val: ../valid/images
test: ../test/images
nc: 6
names: ['Safety_Glasses', 'face_mask', 'gloves', 'goggles', 'lab_coat', 'person']
roboflow:
  workspace: test
  project: labguard
  version: 1
  license: CC BY 4.0
"""


def jpeg_bytes(value: int = 0) -> bytes:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("No se pudo crear la imagen de prueba")
    return encoded.tobytes()


def add_pair(
    archive: zipfile.ZipFile,
    split: str,
    stem: str,
    label: str | None,
    *,
    image: bytes | None = None,
) -> None:
    archive.writestr(f"{split}/images/{stem}.jpg", jpeg_bytes() if image is None else image)
    if label is not None:
        archive.writestr(f"{split}/labels/{stem}.txt", label)


def create_archive(path: Path, populate) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", DATA_YAML)
        archive.writestr("README.dataset.txt", "Dataset de prueba")
        populate(archive)


class AnnotationConversionTest(unittest.TestCase):
    def test_maps_every_source_class_and_marks_glasses_merge(self) -> None:
        self.assertEqual(CLASS_MAPPING, {0: 3, 1: 5, 2: 2, 3: 3, 4: 1, 5: 0})
        for source_class, target_class in CLASS_MAPPING.items():
            with self.subTest(source_class=source_class):
                annotation = parse_annotation_line(f"{source_class} 0.5 0.5 0.2 0.4")

                self.assertEqual(annotation.source_class_name, SOURCE_CLASS_NAMES[source_class])
                self.assertEqual(annotation.target_class_id, target_class)
                self.assertEqual(annotation.source_format, "detection")
                self.assertEqual(annotation.provisional_merge, source_class in {0, 3})

    def test_converts_valid_polygon_to_bounding_box(self) -> None:
        annotation = parse_annotation_line("2 0.2 0.2 0.8 0.2 0.8 0.6 0.2 0.6")

        self.assertEqual(annotation.target_class_id, 2)
        self.assertEqual(annotation.source_format, "segmentation")
        self.assertEqual(annotation.polygon_point_count, 4)
        self.assertAlmostEqual(annotation.bounding_box[0], 0.5)
        self.assertAlmostEqual(annotation.bounding_box[1], 0.4)
        self.assertAlmostEqual(annotation.bounding_box[2], 0.6)
        self.assertAlmostEqual(annotation.bounding_box[3], 0.4)

    def test_rejects_invalid_detection_and_polygon_annotations(self) -> None:
        invalid_lines = (
            "2 0.5 0.5 0 0.2",
            "2 0.5 0.5 1.2 0.2",
            "6 0.5 0.5 0.2 0.2",
            "2 0.1 0.1 0.2 0.2 0.3",
            "2 0.1 0.1 0.2 0.2 0.3 1.1",
            "2 0.1 0.1 0.2 0.2 0.3 0.3",
        )
        for line in invalid_lines:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_annotation_line(line)


class RoboflowArchiveImportTest(unittest.TestCase):
    def test_preserves_originals_splits_provenance_and_visual_review(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            output = root / "staging"

            def populate(archive: zipfile.ZipFile) -> None:
                archive.writestr("unused/", b"")
                add_pair(
                    archive,
                    "train",
                    "Video-One_mp4-0001_jpg.rf." + "a" * 32,
                    "0 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.8 0.2 0.8 0.6 0.2 0.6",
                )
                add_pair(
                    archive,
                    "valid",
                    "Video-One_mp4-0002_jpg.rf." + "b" * 32,
                    "3 0.5 0.5 0.2 0.2",
                )
                add_pair(
                    archive,
                    "test",
                    "Video-Two_mp4-0003_jpg.rf." + "c" * 32,
                    "5 0.5 0.5 0.2 0.2",
                )

            create_archive(archive_path, populate)
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()

            report = import_roboflow_archive(archive_path, output, samples_per_class=1)

            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), archive_hash)
            self.assertEqual(report["summary"]["images"], 3)
            self.assertEqual(report["summary"]["annotations"], 4)
            self.assertEqual(report["formats"]["polygon_to_bounding_box"], 1)
            self.assertEqual(len(report["formats"]["polygon_conversion_records"]), 1)
            self.assertEqual(report["summary"]["provisional_annotations"], 2)
            self.assertEqual(report["target_class_counts"]["4"]["total"], 0)
            self.assertFalse(report["summary"]["ready_for_training"])
            self.assertTrue((output / "source/data.yaml").is_file())
            self.assertTrue((output / "source/train/images").is_dir())
            self.assertTrue((output / "converted/images/val").is_dir())
            self.assertFalse((output / "converted/images/valid").exists())
            self.assertTrue((output / "review/index.html").is_file())
            self.assertEqual(os.stat(output).st_mode & 0o077, 0)
            self.assertIn("PROVISIONAL", (output / "review/index.html").read_text(encoding="utf-8"))
            self.assertIn("POLY-&gt;BOX", (output / "review/index.html").read_text(encoding="utf-8"))

            annotations = [
                json.loads(line)
                for line in (output / "provenance/annotations.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({item["source_class_id"] for item in annotations}, {0, 2, 3, 5})
            self.assertEqual(
                len((output / "review/provisional_safety_goggles.jsonl").read_text(encoding="utf-8").splitlines()),
                2,
            )
            member_lines = (output / "provenance/archive_members.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(member_lines), report["source"]["archive_member_count"])
            self.assertTrue(any(json.loads(line)["member_type"] == "directory" for line in member_lines))

    def test_identifies_exact_duplicates_augmented_variants_and_video_groups(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            output = root / "staging"
            same_image = jpeg_bytes(25)

            def populate(archive: zipfile.ZipFile) -> None:
                canonical = "Shared-Video_mp4-0001_jpg.rf."
                add_pair(archive, "train", canonical + "a" * 32, "5 0.5 0.5 0.2 0.2", image=same_image)
                add_pair(archive, "train", canonical + "b" * 32, "5 0.5 0.5 0.2 0.2", image=same_image)
                add_pair(
                    archive,
                    "valid",
                    canonical + "c" * 32,
                    "2 0.5 0.5 0.2 0.2",
                    image=same_image,
                )
                add_pair(
                    archive,
                    "test",
                    "Other-Video_mp4-0001_jpg.rf." + "d" * 32,
                    "5 0.5 0.5 0.2 0.2",
                    image=jpeg_bytes(75),
                )

            create_archive(archive_path, populate)

            report = import_roboflow_archive(archive_path, output, samples_per_class=0)

            groups = report["groups"]
            self.assertEqual(groups["exact_duplicate_group_count"], 1)
            self.assertEqual(groups["exact_duplicates"][0]["count"], 3)
            self.assertEqual(groups["cross_split_exact_duplicate_group_count"], 1)
            self.assertEqual(groups["conflicting_label_duplicate_group_count"], 1)
            self.assertEqual(groups["augmented_variant_group_count"], 1)
            self.assertEqual(groups["augmented_variants"][0]["count"], 3)
            self.assertEqual(groups["cross_split_augmented_variant_group_count"], 1)
            self.assertEqual(groups["video_origin_group_count"], 2)
            self.assertEqual(groups["cross_split_video_group_count"], 1)
            self.assertTrue(any("duplicadas" in blocker for blocker in report["known_blockers"]))
            self.assertTrue(any("variantes Roboflow" in blocker for blocker in report["known_blockers"]))

    def test_reports_annotation_pairing_and_orphan_errors(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            output = root / "staging"

            def populate(archive: zipfile.ZipFile) -> None:
                add_pair(
                    archive,
                    "train",
                    "Video_mp4-0001_jpg.rf." + "a" * 32,
                    "2 0.5 0.5 0 0.2",
                )
                add_pair(
                    archive,
                    "train",
                    "Video_mp4-0004_jpg.rf." + "d" * 32,
                    "   \n",
                )
                add_pair(
                    archive,
                    "valid",
                    "Video_mp4-0002_jpg.rf." + "b" * 32,
                    None,
                )
                add_pair(
                    archive,
                    "test",
                    "Video_mp4-0003_jpg.rf." + "c" * 32,
                    "5 0.5 0.5 0.2 0.2",
                )
                archive.writestr("test/labels/orphan.txt", "5 0.5 0.5 0.2 0.2")

            create_archive(archive_path, populate)

            report = import_roboflow_archive(archive_path, output, samples_per_class=0)

            self.assertEqual(
                {error["code"] for error in report["errors"]},
                {"invalid_annotation", "missing_label", "orphan_label"},
            )
            self.assertTrue(any("errores" in blocker for blocker in report["known_blockers"]))
            self.assertEqual(report["summary"]["empty_labels"], 1)

    def test_rejects_normalised_and_converted_path_collisions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            normalised_archive = root / "normalised.zip"
            with zipfile.ZipFile(normalised_archive, "w") as archive:
                archive.writestr("data.yaml", DATA_YAML)
                archive.writestr("./data.yaml", DATA_YAML)

            with self.assertRaises(DatasetImportError):
                import_roboflow_archive(normalised_archive, root / "normalised-output")

            converted_archive = root / "converted.zip"
            with zipfile.ZipFile(converted_archive, "w") as archive:
                archive.writestr("data.yaml", DATA_YAML)
                archive.writestr("train/images/sample.jpg", jpeg_bytes())
                archive.writestr("train/images/sample.png", jpeg_bytes())
                archive.writestr("train/labels/sample.txt", "5 0.5 0.5 0.2 0.2")

            with self.assertRaises(DatasetImportError):
                import_roboflow_archive(converted_archive, root / "converted-output")

    def test_rejects_archive_resource_limit_and_removes_partial_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "large.zip"
            output = root / "staging"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("data.yaml", DATA_YAML)
                archive.writestr("oversized.bin", b"x" * 2048)

            with patch("training.import_roboflow.MAX_MEMBER_SIZE", 1024):
                with self.assertRaises(DatasetImportError):
                    import_roboflow_archive(archive_path, output)

            self.assertFalse(output.exists())

    def test_reports_encoded_images_over_the_configured_limit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "images.zip"
            output = root / "staging"

            def populate(archive: zipfile.ZipFile) -> None:
                for index, split in enumerate(("train", "valid", "test")):
                    add_pair(
                        archive,
                        split,
                        f"Video_mp4-000{index}_jpg.rf." + str(index) * 32,
                        "5 0.5 0.5 0.2 0.2",
                    )

            create_archive(archive_path, populate)

            with patch("training.import_roboflow.MAX_ENCODED_IMAGE_SIZE", 1):
                report = import_roboflow_archive(archive_path, output, samples_per_class=0)

            self.assertEqual(sum(error["code"] == "invalid_image" for error in report["errors"]), 3)
            self.assertEqual(report["visual_review"]["sample_count"], 0)

    def test_rejects_non_integer_class_count_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "metadata.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data.yaml", DATA_YAML.replace("nc: 6", "nc: 6.5"))

            with self.assertRaises(DatasetImportError):
                import_roboflow_archive(archive_path, root / "staging")

    def test_rejects_unsafe_zip_paths_and_removes_partial_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            output = root / "staging"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data.yaml", DATA_YAML)
                archive.writestr("../escape.txt", "contenido")

            with self.assertRaises(DatasetImportError):
                import_roboflow_archive(archive_path, output)

            self.assertFalse(output.exists())
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
