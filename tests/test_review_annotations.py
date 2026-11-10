"""Pruebas del editor local de anotaciones YOLO."""

import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import yaml

from training.review_annotations import (
    AnnotationReviewError,
    AnnotationRequestHandler,
    CLASS_NAMES,
    EditorStore,
    RevisionConflict,
    initialise_workspace,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class AnnotationReviewWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "roboflow_import"
        self.work = self.root / "annotation_review"
        (self.source / "converted/images/train").mkdir(parents=True)
        (self.source / "converted/images/val").mkdir(parents=True)
        (self.source / "converted/labels/train").mkdir(parents=True)
        (self.source / "converted/labels/val").mkdir(parents=True)
        train_image = b"test-jpeg-train"
        val_image = b"test-jpeg-val"
        (self.source / "converted/images/train/sample.jpg").write_bytes(train_image)
        (self.source / "converted/images/val/empty.jpg").write_bytes(val_image)
        (self.source / "converted/labels/train/sample.txt").write_text(
            "3 0.3 0.3 0.2 0.2\n3 0.7 0.7 0.2 0.2\n",
            encoding="utf-8",
        )
        (self.source / "converted/labels/val/empty.txt").write_text("", encoding="utf-8")

        images = [
            self.image_record("train", "sample", train_image, annotation_count=2),
            self.image_record("val", "empty", val_image, annotation_count=0),
        ]
        images[1].update(
            canonical_frame="AuditVideo_mp4-0001_jpg",
            video_frame=1,
            video_origin="AuditVideo",
        )
        annotations = [
            self.annotation_record(1, 0, "Safety_Glasses", [0.3, 0.3, 0.2, 0.2]),
            self.annotation_record(2, 3, "goggles", [0.7, 0.7, 0.2, 0.2]),
        ]
        write_jsonl(self.source / "provenance/images.jsonl", images)
        write_jsonl(self.source / "provenance/annotations.jsonl", annotations)
        report = self.source / "reports/import_report.json"
        report.parent.mkdir(parents=True)
        report.write_text('{"schema_version": 1}\n', encoding="utf-8")
        self.source_before = tree_hashes(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def image_record(
        self,
        split: str,
        stem: str,
        image: bytes,
        *,
        annotation_count: int,
    ) -> dict:
        return {
            "annotation_count": annotation_count,
            "canonical_frame": f"Video_mp4-{stem}",
            "converted_image": f"converted/images/{split}/{stem}.jpg",
            "converted_label": f"converted/labels/{split}/{stem}.txt",
            "height": 480,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "source_image": f"{split}/images/{stem}.jpg",
            "source_label": f"{split}/labels/{stem}.txt",
            "source_split": split,
            "target_split": split,
            "video_frame": 1,
            "video_origin": "Video",
            "roboflow_variant": stem,
            "width": 640,
        }

    @staticmethod
    def annotation_record(
        line_number: int,
        source_class_id: int,
        source_class_name: str,
        bbox: list[float],
    ) -> dict:
        return {
            "converted_bounding_box": bbox,
            "line_number": line_number,
            "original_annotation": f"{source_class_id} " + " ".join(map(str, bbox)),
            "provisional_merge": True,
            "source_class_id": source_class_id,
            "source_class_name": source_class_name,
            "source_format": "detection",
            "source_image": "train/images/sample.jpg",
            "source_label": "train/labels/sample.txt",
            "target_class_id": 3,
            "target_class_name": "safety_goggles",
            "target_split": "train",
        }

    def initialise(self) -> EditorStore:
        initialise_workspace(self.source, self.work)
        return EditorStore(self.work)

    def add_audit_image(
        self,
        split: str,
        stem: str,
        *,
        canonical_frame: str,
        video_origin: str,
        video_frame: int,
        class_ids: list[int],
    ) -> None:
        image_bytes = f"image-{stem}".encode()
        image_path = self.source / f"converted/images/{split}/{stem}.jpg"
        label_path = self.source / f"converted/labels/{split}/{stem}.txt"
        image_path.write_bytes(image_bytes)
        label_path.write_text(
            "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in class_ids),
            encoding="utf-8",
        )
        images_path = self.source / "provenance/images.jsonl"
        images = read_jsonl(images_path)
        image = self.image_record(split, stem, image_bytes, annotation_count=len(class_ids))
        image.update(
            canonical_frame=canonical_frame,
            video_origin=video_origin,
            video_frame=video_frame,
            roboflow_variant=f"variant-{stem}",
        )
        images.append(image)
        write_jsonl(images_path, images)

        annotations_path = self.source / "provenance/annotations.jsonl"
        annotations = read_jsonl(annotations_path)
        for line_number, class_id in enumerate(class_ids, start=1):
            source_names = {1: "lab_coat", 3: "goggles", 4: "lab_cap"}
            annotations.append(
                {
                    "converted_bounding_box": [0.5, 0.5, 0.2, 0.2],
                    "line_number": line_number,
                    "original_annotation": f"{class_id} 0.5 0.5 0.2 0.2",
                    "provisional_merge": class_id == 3,
                    "source_class_id": class_id,
                    "source_class_name": source_names[class_id],
                    "source_format": "detection",
                    "source_image": f"{split}/images/{stem}.jpg",
                    "source_label": f"{split}/labels/{stem}.txt",
                    "target_class_id": class_id,
                    "target_class_name": CLASS_NAMES[class_id],
                    "target_split": split,
                }
            )
        write_jsonl(annotations_path, annotations)

    def test_creates_private_independent_copy_with_glasses_provenance(self) -> None:
        manifest = initialise_workspace(self.source, self.work)

        self.assertEqual(len(manifest["images"]), 2)
        self.assertEqual(tuple(manifest["classes"].values()), CLASS_NAMES)
        self.assertEqual(os.stat(self.work).st_mode & 0o077, 0)
        self.assertEqual(tree_hashes(self.source), self.source_before)
        self.assertTrue((self.work / "dataset/images/train/sample.jpg").is_file())
        self.assertTrue((self.work / "dataset/labels/val/empty.txt").is_file())
        self.assertEqual(
            (self.work / "audit_snapshot/annotations.jsonl").read_bytes(),
            (self.source / "provenance/annotations.jsonl").read_bytes(),
        )
        train = next(item for item in manifest["images"] if item["split"] == "train")
        state = EditorStore(self.work).get_image(train["image_id"])
        self.assertEqual(
            [item["origin"]["source_class_name"] for item in state["annotations"]],
            ["Safety_Glasses", "goggles"],
        )
        self.assertTrue(all(item["origin"]["provisional_merge"] for item in state["annotations"]))
        self.assertTrue(all(not item["origin"]["boundary_adjusted"] for item in state["annotations"]))
        self.assertTrue(
            all(item["origin"]["original_bounding_box"] == item["bbox"] for item in state["annotations"])
        )

        with self.assertRaises(AnnotationReviewError):
            initialise_workspace(self.source, self.work)

    def test_saves_create_update_delete_with_history_and_revision_control(self) -> None:
        store = self.initialise()
        train = store.list_images(split="train")["items"][0]
        before = store.get_image(train["image_id"])
        first, second = before["annotations"]
        request = [
            {"id": first["id"], "class_id": 3, "bbox": [0.25, 0.25, 0.2, 0.2]},
            {"id": "browser-temporary-id", "class_id": 4, "bbox": [0.1, 0.1, 0.2, 0.2]},
        ]

        after = store.save(train["image_id"], 0, request)

        self.assertTrue(after["saved"])
        self.assertEqual(after["revision"], 1)
        self.assertEqual(len(after["changes"]["created"]), 1)
        self.assertEqual(len(after["changes"]["updated"]), 1)
        self.assertEqual(after["changes"]["deleted"][0]["id"], second["id"])
        self.assertEqual(after["annotations"][0]["origin"]["source_class_name"], "Safety_Glasses")
        self.assertEqual(after["annotations"][1]["origin"]["kind"], "manual")
        self.assertEqual(
            (self.work / train["working_label"]).read_text(encoding="utf-8"),
            "3 0.25 0.25 0.2 0.2\n4 0.1 0.1 0.2 0.2\n",
        )
        history = store.history(train["image_id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["before"], before["annotations"])
        self.assertEqual(history[0]["after"], after["annotations"])
        self.assertEqual(store.list_images(class_id=4)["total"], 1)
        self.assertEqual(store.list_images(provisional_only=True)["total"], 1)

        with self.assertRaises(RevisionConflict):
            store.save(train["image_id"], 0, request)
        no_change = store.save(
            train["image_id"],
            1,
            [
                {"id": item["id"], "class_id": item["class_id"], "bbox": item["bbox"]}
                for item in after["annotations"]
            ],
        )
        self.assertFalse(no_change["saved"])
        self.assertEqual(no_change["revision"], 1)
        self.assertEqual(len(store.history(train["image_id"])), 1)
        self.assertEqual(tree_hashes(self.source), self.source_before)

    def test_rejects_invalid_boxes_and_exports_versioned_yolo_snapshot(self) -> None:
        store = self.initialise()
        train = store.list_images(split="train")["items"][0]
        state = store.get_image(train["image_id"])
        with self.assertRaises(AnnotationReviewError):
            store.save(
                train["image_id"],
                0,
                [{"id": state["annotations"][0]["id"], "class_id": 4, "bbox": [0.0, 0.0, 0.2, 0.2]}],
            )

        exported = store.export_labels()
        export_root = Path(exported["path"])
        export_manifest = json.loads((export_root / "export_manifest.json").read_text(encoding="utf-8"))
        dataset = yaml.safe_load((export_root / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(export_manifest["image_count"], 2)
        self.assertEqual(export_manifest["class_counts"]["3"]["count"], 2)
        self.assertEqual(dataset["names"], {index: name for index, name in enumerate(CLASS_NAMES)})
        self.assertEqual(Path(dataset["path"]), self.work / "dataset")
        self.assertTrue((export_root / "labels/train/sample.txt").is_file())
        self.assertFalse((export_root / "images").exists())
        self.assertEqual(tree_hashes(self.source), self.source_before)

    def test_editor_html_documents_box_operations_and_review_rules(self) -> None:
        html_path = Path(__file__).parents[1] / "training/review_editor.html"
        html = html_path.read_text(encoding="utf-8")

        for expected in (
            "Agregar caja",
            "Seleccionar / mover",
            "Eliminar anotación",
            "Safety_Glasses",
            "goggles",
            "batas blancas parcialmente visibles",
            "gorros de laboratorio",
            "/api/save",
            "/api/export",
            "/api/audit",
            "Auditar anotaciones",
            "Una etiqueta ausente no demuestra ausencia de EPP",
            "Revisar y decidir en el editor",
            "nunca se copian cajas",
            "function loadCanvasImage",
            "candidate.onload",
            "candidate.onerror",
            "await loadCanvasImage",
            "syncImageSelector(payload)",
            "fuera del filtro",
            "auditError",
            "const target = preferredId || images[0]?.image_id",
            "únicamente una bata tradicional blanca de laboratorio",
            "cleanroom suits",
            "son provisionales",
        ):
            self.assertIn(expected, html)
        navigation = html[html.index("document.querySelectorAll('.audit-review')") :]
        self.assertLess(
            navigation.index("const loaded=await loadImage"),
            navigation.index("document.getElementById('auditPanel').hidden=true"),
        )
        self.assertIn("if(!loaded)", navigation)
        self.assertNotIn("image.onload =", html)

    def test_comparison_and_label_controls_are_client_only(self) -> None:
        html_path = Path(__file__).parents[1] / "training/review_editor.html"
        html = html_path.read_text(encoding="utf-8")

        for expected in (
            'id="originalCanvas"',
            'id="canvas"',
            'id="comparisonView"',
            'id="singleView"',
            'id="showClassNames"',
            "function drawOriginal()",
            "if(showClassNames)",
            "rgba(255,255,255,.95)",
        ):
            self.assertIn(expected, html)
        self.assertNotIn("originalCanvas.addEventListener", html)

        original_draw = html[
            html.index("function drawOriginal()") : html.index("function point(event)")
        ]
        self.assertIn("originalCtx.drawImage(image", original_draw)
        self.assertNotIn("annotations", original_draw)

        view_control = html[
            html.index("function setViewMode(value)") : html.index(
                "document.getElementById('refreshButton')"
            )
        ]
        self.assertNotIn("annotations=", view_control.replace(" ", ""))
        self.assertNotIn("/api/save", view_control)
        self.assertIn("showClassNames=Boolean(value)", view_control.replace(" ", ""))

    def test_audit_prioritizes_related_discrepancies_without_writing(self) -> None:
        self.add_audit_image(
            "train",
            "sample-peer",
            canonical_frame="Video_mp4-sample",
            video_origin="Video",
            video_frame=1,
            class_ids=[1, 3, 4],
        )
        self.add_audit_image(
            "val",
            "audit-before",
            canonical_frame="AuditVideo_mp4-0000_jpg",
            video_origin="AuditVideo",
            video_frame=0,
            class_ids=[1, 3],
        )
        self.add_audit_image(
            "val",
            "audit-after",
            canonical_frame="AuditVideo_mp4-0002_jpg",
            video_origin="AuditVideo",
            video_frame=2,
            class_ids=[1, 3],
        )
        self.source_before = tree_hashes(self.source)
        store = self.initialise()
        workspace_before = tree_hashes(self.work)

        result = store.audit_findings(limit=1000)

        self.assertEqual(tree_hashes(self.work), workspace_before)
        self.assertEqual(tree_hashes(self.source), self.source_before)
        sample = next(
            item for item in store.list_images()["items"] if item["source_image"].endswith("sample.jpg")
        )
        cap_finding = next(
            item
            for item in result["items"]
            if item["image_id"] == sample["image_id"] and item["class_id"] == 4
        )
        self.assertEqual(cap_finding["priority"], "high")
        self.assertTrue(cap_finding["requires_human_review"])
        self.assertIsNone(cap_finding["automatic_action"])
        self.assertNotIn("bbox", cap_finding)
        self.assertEqual(cap_finding["signals"][0]["relation"], "augmented_variant")
        self.assertEqual(cap_finding["signals"][0]["related_images"][0]["class_count"], 1)

        empty = next(
            item for item in store.list_images(split="val")["items"] if item["source_image"].endswith("empty.jpg")
        )
        coat_finding = next(
            item
            for item in result["items"]
            if item["image_id"] == empty["image_id"] and item["class_id"] == 1
        )
        self.assertEqual(coat_finding["signals"][0]["relation"], "video_neighbor")
        self.assertEqual(len(coat_finding["signals"][0]["related_images"]), 2)
        caps = store.audit_findings(class_id=4, relation="augmented_variant", limit=1000)
        self.assertTrue(caps["items"])
        self.assertTrue(all(item["class_id"] == 4 for item in caps["items"]))
        self.assertTrue(
            all(
                signal["relation"] == "augmented_variant"
                for item in caps["items"]
                for signal in item["signals"]
            )
        )

        state = store.get_image(sample["image_id"])
        request = [
            {"id": item["id"], "class_id": item["class_id"], "bbox": item["bbox"]}
            for item in state["annotations"]
        ]
        request.append({"id": "manual-cap", "class_id": 4, "bbox": [0.1, 0.1, 0.2, 0.2]})
        store.save(sample["image_id"], 0, request)
        refreshed = store.audit_findings(class_id=4, limit=1000)
        self.assertFalse(any(item["image_id"] == sample["image_id"] for item in refreshed["items"]))

        with self.assertRaises(AnnotationReviewError):
            store.audit_findings(relation="copy_boxes")

    def test_filtered_audit_finding_resolves_annotation_and_image_apis(self) -> None:
        self.add_audit_image(
            "train",
            "sample-peer",
            canonical_frame="Video_mp4-sample",
            video_origin="Video",
            video_frame=1,
            class_ids=[1, 3, 4],
        )
        store = self.initialise()
        handler = type("FilteredAuditHandler", (AnnotationRequestHandler,), {"store": store})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            address = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{address}/api/images?split=val&class_id=4", timeout=5) as response:
                filtered = json.load(response)
            self.assertEqual(filtered["items"], [])

            with urlopen(f"{address}/api/audit?class_id=4", timeout=5) as response:
                audit = json.load(response)
            finding = next(
                item for item in audit["items"] if item["source_image"].endswith("sample.jpg")
            )
            self.assertNotIn(finding["image_id"], {item["image_id"] for item in filtered["items"]})

            with urlopen(
                f"{address}/api/annotation?id={finding['image_id']}", timeout=5
            ) as response:
                annotation = json.load(response)
            self.assertEqual(annotation["image_id"], finding["image_id"])
            self.assertEqual(annotation["revision"], finding["revision"])
            self.assertEqual(len(annotation["annotations"]), 2)

            with urlopen(f"{address}/api/image?id={finding['image_id']}", timeout=5) as response:
                image_bytes = response.read()
                content_type = response.headers["Content-Type"]
            self.assertEqual(image_bytes, b"test-jpeg-train")
            self.assertEqual(content_type, "image/jpeg")

            with self.assertRaises(HTTPError) as context:
                urlopen(f"{address}/api/annotation?id=unknown", timeout=5)
            self.assertEqual(context.exception.code, 400)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_local_http_server_serves_editor_and_image_catalog(self) -> None:
        store = self.initialise()
        workspace_before = tree_hashes(self.work)
        handler = type("TestAnnotationHandler", (AnnotationRequestHandler,), {"store": store})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            address = f"http://127.0.0.1:{server.server_port}"
            with urlopen(address, timeout=5) as response:
                self.assertIn("LabGuard · Revisión YOLO", response.read().decode("utf-8"))
            with urlopen(f"{address}/api/images?limit=3000", timeout=5) as response:
                catalog = json.load(response)
            with urlopen(f"{address}/api/audit?class_id=4", timeout=5) as response:
                audit = json.load(response)
            image_id = catalog["items"][0]["image_id"]
            with urlopen(f"{address}/api/annotation?id={image_id}", timeout=5) as response:
                annotation = json.load(response)
            self.assertEqual(catalog["total"], 2)
            self.assertEqual(len(catalog["items"]), 2)
            self.assertTrue(audit["notice"].startswith("Hallazgos heurísticos"))
            self.assertEqual(annotation["revision"], 0)
            self.assertEqual(tree_hashes(self.work), workspace_before)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
