"""Editor web local para revisar una copia de trabajo de anotaciones YOLO."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / ".labguard-work" / "roboflow_import"
DEFAULT_WORK = PROJECT_ROOT / ".labguard-work" / "annotation_review"
EDITOR_HTML = Path(__file__).with_name("review_editor.html")
CLASS_NAMES = (
    "person",
    "lab_coat",
    "protective_gloves",
    "safety_goggles",
    "lab_cap",
    "face_mask",
)
SPLITS = ("train", "val", "test")
MAX_REQUEST_BYTES = 4 * 1024 * 1024
AUDIT_RELATIONS = ("augmented_variant", "video_neighbor")
AUDIT_PRIORITIES = ("high", "medium", "low")
AUDIT_CLASS_BOOST = {4: 25, 3: 15, 1: 12}
VIDEO_NEIGHBOR_MAX_GAP = 10


class AnnotationReviewError(RuntimeError):
    """Indica que el workspace o una correccion no son validos."""


class RevisionConflict(AnnotationReviewError):
    """Indica que otra operacion guardo la imagen antes que el cliente."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AnnotationReviewError(f"No se pudo leer {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnnotationReviewError(f"JSONL invalido en {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise AnnotationReviewError(f"Registro no valido en {path}:{line_number}")
        records.append(record)
    return records


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        pending = memoryview(line.encode("utf-8"))
        while pending:
            pending = pending[os.write(descriptor, pending) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalise_bbox_values(raw_bbox: Any) -> list[float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise AnnotationReviewError("bbox debe contener [x_center, y_center, width, height]")
    if any(isinstance(value, bool) for value in raw_bbox):
        raise AnnotationReviewError("bbox solo admite numeros")
    try:
        x_center, y_center, width, height = (float(value) for value in raw_bbox)
    except (TypeError, ValueError) as error:
        raise AnnotationReviewError("bbox solo admite numeros") from error
    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        raise AnnotationReviewError("bbox debe contener numeros finitos")
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise AnnotationReviewError("el centro de la caja debe estar entre 0 y 1")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise AnnotationReviewError("ancho y alto deben estar entre 0 (exclusivo) y 1")
    return [x_center, y_center, width, height]


def _normalise_bbox(raw_bbox: Any) -> list[float]:
    x_center, y_center, width, height = _normalise_bbox_values(raw_bbox)
    if (
        x_center - width / 2 < -1e-8
        or x_center + width / 2 > 1.0 + 1e-8
        or y_center - height / 2 < -1e-8
        or y_center + height / 2 > 1.0 + 1e-8
    ):
        raise AnnotationReviewError("la caja debe permanecer dentro de la imagen")
    return [x_center, y_center, width, height]


def _normalise_class_id(raw_class_id: Any) -> int:
    if isinstance(raw_class_id, bool) or not isinstance(raw_class_id, int):
        raise AnnotationReviewError("class_id debe ser un entero")
    if raw_class_id not in range(len(CLASS_NAMES)):
        raise AnnotationReviewError(f"class_id debe estar entre 0 y {len(CLASS_NAMES) - 1}")
    return raw_class_id


def _format_coordinate(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def _yolo_text(annotations: Iterable[dict[str, Any]]) -> str:
    lines = []
    for annotation in annotations:
        coordinates = " ".join(_format_coordinate(value) for value in annotation["bbox"])
        lines.append(f"{annotation['class_id']} {coordinates}\n")
    return "".join(lines)


def _safe_relative_path(raw_path: Any) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise AnnotationReviewError("La procedencia contiene una ruta vacia")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or "\\" in raw_path:
        raise AnnotationReviewError(f"Ruta de procedencia insegura: {raw_path!r}")
    return path


def _state_annotation(provenance: dict[str, Any]) -> dict[str, Any]:
    source_label = str(provenance["source_label"])
    line_number = int(provenance["line_number"])
    annotation_id = hashlib.sha256(f"{source_label}:{line_number}".encode()).hexdigest()[:24]
    class_id = _normalise_class_id(provenance["target_class_id"])
    bbox = _normalise_bbox(provenance["converted_bounding_box"])
    original_bbox = _normalise_bbox_values(provenance.get("original_bounding_box", bbox))
    return {
        "id": annotation_id,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "bbox": bbox,
        "origin": {
            "kind": "roboflow_import",
            "source_class_id": int(provenance["source_class_id"]),
            "source_class_name": str(provenance["source_class_name"]),
            "source_label": source_label,
            "line_number": line_number,
            "source_format": str(provenance["source_format"]),
            "provisional_merge": bool(provenance["provisional_merge"]),
            "boundary_adjusted": bool(provenance.get("boundary_adjusted", False)),
            "original_bounding_box": original_bbox,
            "original_annotation": str(provenance["original_annotation"]),
        },
    }


def _workspace_source_files(source_root: Path) -> tuple[Path, Path, Path]:
    image_provenance = source_root / "provenance" / "images.jsonl"
    annotation_provenance = source_root / "provenance" / "annotations.jsonl"
    import_report = source_root / "reports" / "import_report.json"
    for path in (image_provenance, annotation_provenance, import_report):
        if not path.is_file():
            raise AnnotationReviewError(f"Falta el artefacto auditado: {path}")
    return image_provenance, annotation_provenance, import_report


def initialise_workspace(source_root: str | Path, work_root: str | Path) -> dict[str, Any]:
    """Crea de forma atomica una copia editable sin alterar el staging auditado."""

    source = Path(source_root).expanduser().resolve()
    work = Path(work_root).expanduser().resolve()
    image_provenance_path, annotation_provenance_path, report_path = _workspace_source_files(source)
    if work.exists():
        raise AnnotationReviewError(f"La copia de trabajo ya existe: {work}")
    if work == source or source in work.parents or work in source.parents:
        raise AnnotationReviewError("La copia de trabajo debe estar separada del staging auditado")

    previous_umask = os.umask(0o077)
    temporary = work.parent / f".{work.name}.building-{uuid.uuid4().hex}"
    try:
        work.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        image_provenance = _read_jsonl(image_provenance_path)
        annotation_provenance = _read_jsonl(annotation_provenance_path)
        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotation_provenance:
            annotations_by_image[str(annotation["source_image"])].append(annotation)

        audit_snapshot = temporary / "audit_snapshot"
        audit_snapshot.mkdir()
        shutil.copy2(image_provenance_path, audit_snapshot / "images.jsonl")
        shutil.copy2(annotation_provenance_path, audit_snapshot / "annotations.jsonl")
        shutil.copy2(report_path, audit_snapshot / "import_report.json")

        manifest_images = []
        seen_ids: set[str] = set()
        for image_record in image_provenance:
            source_image_key = str(image_record["source_image"])
            target_split = str(image_record["target_split"])
            if target_split not in SPLITS:
                raise AnnotationReviewError(f"Split no valido en procedencia: {target_split}")
            converted_image_relative = _safe_relative_path(image_record["converted_image"])
            converted_label_relative = _safe_relative_path(image_record["converted_label"])
            source_image_path = source.joinpath(*converted_image_relative.parts)
            source_label_path = source.joinpath(*converted_label_relative.parts)
            if not source_image_path.is_file() or not source_label_path.is_file():
                raise AnnotationReviewError(f"Falta imagen o etiqueta convertida para {source_image_key}")

            image_tail = converted_image_relative.relative_to(
                PurePosixPath("converted", "images", target_split)
            )
            label_tail = converted_label_relative.relative_to(
                PurePosixPath("converted", "labels", target_split)
            )
            working_image_relative = PurePosixPath("dataset", "images", target_split, *image_tail.parts)
            working_label_relative = PurePosixPath("dataset", "labels", target_split, *label_tail.parts)
            state_relative = PurePosixPath("state", target_split, *label_tail.with_suffix(".json").parts)
            working_image = temporary.joinpath(*working_image_relative.parts)
            working_label = temporary.joinpath(*working_label_relative.parts)
            state_path = temporary.joinpath(*state_relative.parts)
            working_image.parent.mkdir(parents=True, exist_ok=True)
            working_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image_path, working_image)

            annotations = [
                _state_annotation(annotation)
                for annotation in sorted(
                    annotations_by_image[source_image_key],
                    key=lambda item: int(item["line_number"]),
                )
            ]
            image_id = hashlib.sha256(source_image_key.encode()).hexdigest()
            if image_id in seen_ids:
                raise AnnotationReviewError(f"ID de imagen duplicado para {source_image_key}")
            seen_ids.add(image_id)
            state = {
                "schema_version": 1,
                "image_id": image_id,
                "revision": 0,
                "last_modified_utc": None,
                "image": {
                    "source_image": source_image_key,
                    "source_split": str(image_record["source_split"]),
                    "target_split": target_split,
                    "working_image": working_image_relative.as_posix(),
                    "working_label": working_label_relative.as_posix(),
                    "width": int(image_record["width"]),
                    "height": int(image_record["height"]),
                    "image_sha256": str(image_record["image_sha256"]),
                    "canonical_frame": image_record.get("canonical_frame"),
                    "video_origin": image_record.get("video_origin"),
                    "video_frame": image_record.get("video_frame"),
                },
                "annotations": annotations,
            }
            _atomic_json(state_path, state)
            _atomic_text(working_label, _yolo_text(annotations))
            manifest_images.append(_manifest_entry(state, state_relative))

        dataset_yaml = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        (temporary / "dataset" / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "source": {
                "audited_staging": str(source),
                "import_report_sha256": _sha256_file(report_path),
                "image_provenance_sha256": _sha256_file(image_provenance_path),
                "annotation_provenance_sha256": _sha256_file(annotation_provenance_path),
            },
            "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "images": manifest_images,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        (temporary / "history").mkdir()
        (temporary / "exports").mkdir()
        os.replace(temporary, work)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        os.umask(previous_umask)


def _manifest_entry(state: dict[str, Any], state_relative: PurePosixPath | str) -> dict[str, Any]:
    annotations = state["annotations"]
    return {
        "image_id": state["image_id"],
        "source_image": state["image"]["source_image"],
        "split": state["image"]["target_split"],
        "working_image": state["image"]["working_image"],
        "working_label": state["image"]["working_label"],
        "state_path": str(state_relative),
        "width": state["image"]["width"],
        "height": state["image"]["height"],
        "video_origin": state["image"].get("video_origin"),
        "video_frame": state["image"].get("video_frame"),
        "revision": state["revision"],
        "annotation_count": len(annotations),
        "class_ids": sorted({annotation["class_id"] for annotation in annotations}),
        "provisional_count": sum(
            bool(annotation.get("origin", {}).get("provisional_merge")) for annotation in annotations
        ),
    }


class EditorStore:
    """Mantiene estados editables, etiquetas YOLO e historial local."""

    def __init__(self, work_root: str | Path) -> None:
        self.root = Path(work_root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise AnnotationReviewError(f"No existe manifest.json en {self.root}")
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationReviewError(f"No se pudo leer el manifest: {error}") from error
        if tuple(self.manifest.get("classes", {}).values()) != CLASS_NAMES:
            raise AnnotationReviewError("El workspace no usa las seis clases LabGuard esperadas")
        self._images = {item["image_id"]: item for item in self.manifest["images"]}
        if len(self._images) != len(self.manifest["images"]):
            raise AnnotationReviewError("El manifest contiene IDs de imagen duplicados")
        self._lock = threading.RLock()
        audit_records = _read_jsonl(self.root / "audit_snapshot" / "images.jsonl")
        self._audit_metadata = {str(item["source_image"]): item for item in audit_records}
        if len(self._audit_metadata) != len(audit_records):
            raise AnnotationReviewError("El snapshot de auditoria contiene imagenes duplicadas")
        self._audit_cache: dict[str, Any] | None = None

    def _entry(self, image_id: str) -> dict[str, Any]:
        try:
            return self._images[image_id]
        except KeyError as error:
            raise AnnotationReviewError("Imagen desconocida") from error

    def _inside_root(self, relative: str) -> Path:
        path = self.root.joinpath(*_safe_relative_path(relative).parts).resolve()
        if self.root not in path.parents:
            raise AnnotationReviewError("Ruta fuera de la copia de trabajo")
        return path

    def _load_state(self, entry: dict[str, Any]) -> dict[str, Any]:
        path = self._inside_root(entry["state_path"])
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationReviewError(f"Estado de imagen invalido: {error}") from error
        return state

    def get_image(self, image_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_state(self._entry(image_id))
            return {"classes": list(enumerate(CLASS_NAMES)), **state}

    def image_path(self, image_id: str) -> Path:
        with self._lock:
            entry = self._entry(image_id)
            path = self._inside_root(entry["working_image"])
            if not path.is_file():
                raise AnnotationReviewError("No existe la imagen de trabajo")
            return path

    def list_images(
        self,
        *,
        split: str = "all",
        query: str = "",
        class_id: int | None = None,
        provisional_only: bool = False,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        if split != "all" and split not in SPLITS:
            raise AnnotationReviewError("Filtro de split no valido")
        if class_id is not None:
            _normalise_class_id(class_id)
        query = query.casefold().strip()
        items = []
        for item in self.manifest["images"]:
            if split != "all" and item["split"] != split:
                continue
            if query and query not in item["source_image"].casefold():
                continue
            if class_id is not None and class_id not in item["class_ids"]:
                continue
            if provisional_only and not item["provisional_count"]:
                continue
            items.append(item)
        items.sort(key=lambda item: (SPLITS.index(item["split"]), item["source_image"]))
        return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}

    @staticmethod
    def _audit_peer(item: dict[str, Any], class_id: int) -> dict[str, Any]:
        return {
            "image_id": item["image_id"],
            "source_image": item["source_image"],
            "split": item["split"],
            "revision": item["revision"],
            "roboflow_variant": item["roboflow_variant"],
            "video_frame": item["video_frame"],
            "class_count": item["class_counts"].get(class_id, 0),
            "class_counts": {
                str(key): value for key, value in sorted(item["class_counts"].items())
            },
        }

    def _build_audit(self) -> dict[str, Any]:
        images = []
        canonical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        video_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self.manifest["images"]:
            state = self._load_state(entry)
            metadata = self._audit_metadata.get(entry["source_image"], {})
            counts = Counter(annotation["class_id"] for annotation in state["annotations"])
            image = {
                "image_id": entry["image_id"],
                "source_image": entry["source_image"],
                "split": entry["split"],
                "revision": state["revision"],
                "class_counts": dict(counts),
                "canonical_frame": state["image"].get("canonical_frame")
                or metadata.get("canonical_frame"),
                "roboflow_variant": metadata.get("roboflow_variant"),
                "video_origin": state["image"].get("video_origin")
                or metadata.get("video_origin"),
                "video_frame": state["image"].get("video_frame"),
            }
            images.append(image)
            if image["canonical_frame"]:
                canonical_groups[str(image["canonical_frame"])].append(image)
            if image["video_origin"] and isinstance(image["video_frame"], int):
                video_groups[str(image["video_origin"])].append(image)

        findings: dict[tuple[str, int], dict[str, Any]] = {}

        def add_signal(
            image: dict[str, Any],
            class_id: int,
            relation: str,
            group_key: str,
            expected_count: int,
            peers: list[dict[str, Any]],
            signal_score: int,
        ) -> None:
            key = (image["image_id"], class_id)
            observed_count = image["class_counts"].get(class_id, 0)
            finding = findings.setdefault(
                key,
                {
                    "finding_id": hashlib.sha256(
                        f"{image['image_id']}:{class_id}".encode()
                    ).hexdigest()[:24],
                    "image_id": image["image_id"],
                    "source_image": image["source_image"],
                    "split": image["split"],
                    "revision": image["revision"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "observed_count": observed_count,
                    "canonical_frame": image["canonical_frame"],
                    "roboflow_variant": image["roboflow_variant"],
                    "video_origin": image["video_origin"],
                    "video_frame": image["video_frame"],
                    "signals": [],
                    "signal_score": 0,
                },
            )
            finding["signal_score"] = max(finding["signal_score"], signal_score)
            finding["signals"].append(
                {
                    "relation": relation,
                    "group_key": group_key,
                    "observed_count": observed_count,
                    "peer_count": expected_count,
                    "related_images": [
                        self._audit_peer(peer, class_id)
                        for peer in sorted(peers, key=lambda item: item["source_image"])[:6]
                    ],
                }
            )

        variant_groups = [group for group in canonical_groups.values() if len(group) > 1]
        variant_discrepancy_groups = 0
        for group in variant_groups:
            signatures = {tuple(sorted(item["class_counts"].items())) for item in group}
            if len(signatures) > 1:
                variant_discrepancy_groups += 1
            for class_id in range(len(CLASS_NAMES)):
                maximum = max(item["class_counts"].get(class_id, 0) for item in group)
                if maximum == 0:
                    continue
                for image in group:
                    observed = image["class_counts"].get(class_id, 0)
                    if observed >= maximum:
                        continue
                    peers = [
                        peer
                        for peer in group
                        if peer["image_id"] != image["image_id"]
                        and peer["class_counts"].get(class_id, 0) > observed
                    ]
                    support_ratio = len(peers) / max(1, len(group) - 1)
                    score = (
                        45
                        + AUDIT_CLASS_BOOST.get(class_id, 0)
                        + (15 if observed == 0 else 5)
                        + round(15 * support_ratio)
                    )
                    add_signal(
                        image,
                        class_id,
                        "augmented_variant",
                        str(image["canonical_frame"]),
                        maximum,
                        peers,
                        score,
                    )

        video_comparisons = 0
        for video_origin, group in video_groups.items():
            by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for image in group:
                by_frame[int(image["video_frame"])].append(image)
            frames = sorted(by_frame)
            for index in range(1, len(frames) - 1):
                previous_frame, frame, next_frame = frames[index - 1 : index + 2]
                if (
                    frame - previous_frame > VIDEO_NEIGHBOR_MAX_GAP
                    or next_frame - frame > VIDEO_NEIGHBOR_MAX_GAP
                ):
                    continue
                video_comparisons += 1
                current = by_frame[frame]
                previous = by_frame[previous_frame]
                following = by_frame[next_frame]
                for class_id in range(len(CLASS_NAMES)):
                    current_max = max(
                        item["class_counts"].get(class_id, 0) for item in current
                    )
                    previous_max = max(
                        item["class_counts"].get(class_id, 0) for item in previous
                    )
                    following_max = max(
                        item["class_counts"].get(class_id, 0) for item in following
                    )
                    expected = min(previous_max, following_max)
                    if expected <= current_max:
                        continue
                    for image in current:
                        observed = image["class_counts"].get(class_id, 0)
                        if observed >= expected:
                            continue
                        peers = [
                            peer
                            for peer in previous + following
                            if peer["class_counts"].get(class_id, 0) > observed
                        ]
                        score = (
                            25
                            + AUDIT_CLASS_BOOST.get(class_id, 0)
                            + (15 if observed == 0 else 5)
                            + 15
                        )
                        add_signal(
                            image,
                            class_id,
                            "video_neighbor",
                            video_origin,
                            expected,
                            peers,
                            score,
                        )

        prepared = []
        for finding in findings.values():
            score = min(100, finding.pop("signal_score") + 10 * (len(finding["signals"]) - 1))
            priority = "high" if score >= 80 else "medium" if score >= 60 else "low"
            prepared.append(
                {
                    **finding,
                    "score": score,
                    "priority": priority,
                    "requires_human_review": True,
                    "automatic_action": None,
                }
            )
        prepared.sort(
            key=lambda item: (
                -item["score"],
                0 if item["class_id"] in (4, 1, 3) else 1,
                item["source_image"],
                item["class_id"],
            )
        )
        return {
            "findings": prepared,
            "groups": {
                "canonical_frame_groups": len(canonical_groups),
                "augmented_variant_groups": len(variant_groups),
                "variant_discrepancy_groups": variant_discrepancy_groups,
                "video_origin_groups": len(video_groups),
                "video_neighbor_comparisons": video_comparisons,
            },
        }

    def audit_findings(
        self,
        *,
        split: str = "all",
        query: str = "",
        class_id: int | None = None,
        priority: str = "all",
        relation: str = "all",
        offset: int = 0,
        limit: int = 250,
    ) -> dict[str, Any]:
        """Compara grupos relacionados sin escribir ni proponer cajas automaticas."""

        if split != "all" and split not in SPLITS:
            raise AnnotationReviewError("Filtro de split no valido")
        if class_id is not None:
            _normalise_class_id(class_id)
        if priority != "all" and priority not in AUDIT_PRIORITIES:
            raise AnnotationReviewError("Prioridad de auditoria no valida")
        if relation != "all" and relation not in AUDIT_RELATIONS:
            raise AnnotationReviewError("Relacion de auditoria no valida")
        query = query.casefold().strip()
        with self._lock:
            if self._audit_cache is None:
                self._audit_cache = self._build_audit()
            all_findings = self._audit_cache["findings"]
            findings = []
            for finding in all_findings:
                if split != "all" and finding["split"] != split:
                    continue
                if class_id is not None and finding["class_id"] != class_id:
                    continue
                if priority != "all" and finding["priority"] != priority:
                    continue
                if relation != "all" and not any(
                    signal["relation"] == relation for signal in finding["signals"]
                ):
                    continue
                searchable = " ".join(
                    str(finding.get(key) or "")
                    for key in ("source_image", "canonical_frame", "video_origin")
                ).casefold()
                if query and query not in searchable:
                    continue
                findings.append(finding)
            priority_counts = Counter(item["priority"] for item in all_findings)
            class_counts = Counter(item["class_id"] for item in all_findings)
            return {
                "classes": list(enumerate(CLASS_NAMES)),
                "notice": (
                    "Hallazgos heurísticos: una etiqueta ausente no demuestra ausencia de EPP. "
                    "Revise visualmente y edite solo con aprobación humana."
                ),
                "groups": self._audit_cache["groups"],
                "summary": {
                    "total_findings": len(all_findings),
                    "filtered_findings": len(findings),
                    "priority_counts": {
                        name: priority_counts[name] for name in AUDIT_PRIORITIES
                    },
                    "class_counts": {
                        str(index): class_counts[index] for index in range(len(CLASS_NAMES))
                    },
                },
                "offset": offset,
                "items": findings[offset : offset + limit],
            }

    def save(self, image_id: str, expected_revision: int, raw_annotations: Any) -> dict[str, Any]:
        if not isinstance(raw_annotations, list):
            raise AnnotationReviewError("annotations debe ser una lista")
        with self._lock:
            entry = self._entry(image_id)
            before = self._load_state(entry)
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
                raise AnnotationReviewError("expected_revision debe ser entero")
            if expected_revision != before["revision"]:
                raise RevisionConflict("La imagen cambio; recargue antes de guardar")

            existing = {annotation["id"]: annotation for annotation in before["annotations"]}
            seen_ids: set[str] = set()
            now = _utc_now()
            annotations = []
            for raw in raw_annotations:
                if not isinstance(raw, dict):
                    raise AnnotationReviewError("Cada anotacion debe ser un objeto")
                class_id = _normalise_class_id(raw.get("class_id"))
                bbox = _normalise_bbox(raw.get("bbox"))
                raw_id = raw.get("id")
                if isinstance(raw_id, str) and raw_id in existing:
                    annotation_id = raw_id
                    origin = existing[annotation_id]["origin"]
                else:
                    annotation_id = uuid.uuid4().hex
                    origin = {"kind": "manual", "created_at_utc": now}
                if annotation_id in seen_ids:
                    raise AnnotationReviewError("ID de anotacion duplicado")
                seen_ids.add(annotation_id)
                annotations.append(
                    {
                        "id": annotation_id,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "bbox": bbox,
                        "origin": origin,
                    }
                )

            after = {
                **before,
                "revision": before["revision"] + 1,
                "last_modified_utc": now,
                "annotations": annotations,
            }
            changes = _describe_changes(before["annotations"], annotations)
            if not any(changes.values()):
                return {"saved": False, "changes": changes, **before}

            state_path = self._inside_root(entry["state_path"])
            label_path = self._inside_root(entry["working_label"])
            _atomic_json(state_path, after)
            _atomic_text(label_path, _yolo_text(annotations))
            history_event = {
                "event_id": uuid.uuid4().hex,
                "timestamp_utc": now,
                "image_id": image_id,
                "source_image": entry["source_image"],
                "revision_before": before["revision"],
                "revision_after": after["revision"],
                "changes": changes,
                "before": before["annotations"],
                "after": annotations,
            }
            _append_jsonl(self.root / "history" / "corrections.jsonl", history_event)
            new_entry = _manifest_entry(after, entry["state_path"])
            entry.clear()
            entry.update(new_entry)
            _atomic_json(self.manifest_path, self.manifest)
            self._audit_cache = None
            return {"saved": True, "changes": changes, "classes": list(enumerate(CLASS_NAMES)), **after}

    def history(self, image_id: str, limit: int = 100) -> list[dict[str, Any]]:
        self._entry(image_id)
        path = self.root / "history" / "corrections.jsonl"
        if not path.exists():
            return []
        events = [event for event in _read_jsonl(path) if event.get("image_id") == image_id]
        return events[-limit:]

    def export_labels(self) -> dict[str, Any]:
        with self._lock:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            export_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
            export_root = self.root / "exports" / export_id
            labels_root = export_root / "labels"
            labels_root.mkdir(parents=True)
            counts = Counter()
            label_hashes = []
            for item in self.manifest["images"]:
                source_label = self._inside_root(item["working_label"])
                relative = PurePosixPath(item["working_label"]).relative_to(
                    PurePosixPath("dataset", "labels")
                )
                destination = labels_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_label, destination)
                label_hashes.append(
                    {
                        "label": f"labels/{relative.as_posix()}",
                        "sha256": _sha256_file(destination),
                    }
                )
                state = self._load_state(item)
                for annotation in state["annotations"]:
                    counts[annotation["class_id"]] += 1

            dataset_root = self.root / "dataset"
            dataset_yaml = {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            }
            (export_root / "dataset.yaml").write_text(
                yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            export_manifest = {
                "schema_version": 1,
                "export_id": export_id,
                "created_at_utc": _utc_now(),
                "workspace": str(self.root),
                "source_manifest_sha256": _sha256_file(self.manifest_path),
                "history_sha256": (
                    _sha256_file(self.root / "history" / "corrections.jsonl")
                    if (self.root / "history" / "corrections.jsonl").is_file()
                    else None
                ),
                "image_count": len(self.manifest["images"]),
                "class_counts": {
                    str(index): {"name": name, "count": counts[index]}
                    for index, name in enumerate(CLASS_NAMES)
                },
                "split_counts": dict(Counter(item["split"] for item in self.manifest["images"])),
                "labels": label_hashes,
                "note": "Solo etiquetas. Las imagenes permanecen en la copia de trabajo y no se redistribuyeron.",
            }
            _atomic_json(export_root / "export_manifest.json", export_manifest)
            return {
                "export_id": export_id,
                "path": str(export_root),
                "image_count": export_manifest["image_count"],
                "class_counts": export_manifest["class_counts"],
            }


def _describe_changes(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    before_by_id = {annotation["id"]: annotation for annotation in before}
    after_by_id = {annotation["id"]: annotation for annotation in after}
    created = [after_by_id[key] for key in after_by_id.keys() - before_by_id.keys()]
    deleted = [before_by_id[key] for key in before_by_id.keys() - after_by_id.keys()]
    updated = []
    for annotation_id in before_by_id.keys() & after_by_id.keys():
        old = before_by_id[annotation_id]
        new = after_by_id[annotation_id]
        if old["class_id"] != new["class_id"] or old["bbox"] != new["bbox"]:
            updated.append({"id": annotation_id, "before": old, "after": new})
    return {"created": created, "updated": updated, "deleted": deleted}


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    """API local y entrega del editor HTML."""

    store: EditorStore

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, error: Exception) -> None:
        status = HTTPStatus.CONFLICT if isinstance(error, RevisionConflict) else HTTPStatus.BAD_REQUEST
        self._json({"error": str(error)}, status)

    def _request_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise AnnotationReviewError("Content-Length no valido") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise AnnotationReviewError("Tamano de solicitud no valido")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise AnnotationReviewError("JSON no valido") from error
        if not isinstance(payload, dict):
            raise AnnotationReviewError("La solicitud debe ser un objeto JSON")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                content = EDITOR_HTML.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            if parsed.path == "/api/images":
                raw_class = query.get("class_id", [""])[0]
                class_id = int(raw_class) if raw_class else None
                self._json(
                    self.store.list_images(
                        split=query.get("split", ["all"])[0],
                        query=query.get("query", [""])[0],
                        class_id=class_id,
                        provisional_only=query.get("provisional", ["0"])[0] == "1",
                        offset=max(0, int(query.get("offset", ["0"])[0])),
                        limit=min(5000, max(1, int(query.get("limit", ["500"])[0]))),
                    )
                )
                return
            if parsed.path == "/api/annotation":
                self._json(self.store.get_image(query.get("id", [""])[0]))
                return
            if parsed.path == "/api/audit":
                raw_class = query.get("class_id", [""])[0]
                class_id = int(raw_class) if raw_class else None
                self._json(
                    self.store.audit_findings(
                        split=query.get("split", ["all"])[0],
                        query=query.get("query", [""])[0],
                        class_id=class_id,
                        priority=query.get("priority", ["all"])[0],
                        relation=query.get("relation", ["all"])[0],
                        offset=max(0, int(query.get("offset", ["0"])[0])),
                        limit=min(2000, max(1, int(query.get("limit", ["250"])[0]))),
                    )
                )
                return
            if parsed.path == "/api/history":
                self._json({"events": self.store.history(query.get("id", [""])[0])})
                return
            if parsed.path == "/api/image":
                image_path = self.store.image_path(query.get("id", [""])[0])
                content = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(image_path.name)[0] or "image/jpeg")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            self._json({"error": "Ruta no encontrada"}, HTTPStatus.NOT_FOUND)
        except (AnnotationReviewError, OSError, ValueError) as error:
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._request_json()
            if self.path == "/api/save":
                self._json(
                    self.store.save(
                        str(payload.get("image_id", "")),
                        payload.get("expected_revision"),
                        payload.get("annotations"),
                    )
                )
                return
            if self.path == "/api/export":
                self._json(self.store.export_labels(), HTTPStatus.CREATED)
                return
            self._json({"error": "Ruta no encontrada"}, HTTPStatus.NOT_FOUND)
        except (AnnotationReviewError, OSError, ValueError) as error:
            self._error(error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        print("Error: --port debe estar entre 1 y 65535")
        return 2
    try:
        if not args.work.expanduser().resolve().exists():
            manifest = initialise_workspace(args.source, args.work)
            print(f"Copia de trabajo creada con {len(manifest['images'])} imagenes: {args.work.resolve()}")
        store = EditorStore(args.work)
    except (AnnotationReviewError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    if args.prepare_only:
        print("Preparacion completada. El staging auditado no fue modificado.")
        return 0

    handler = type("LabGuardAnnotationHandler", (AnnotationRequestHandler,), {"store": store})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Editor local: {url}")
    print("Detener con Ctrl+C. Los cambios se guardan solo en la copia de trabajo.")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEditor detenido.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
