"""Importa un ZIP Roboflow a un staging auditable sin alterar el archivo fuente."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Iterable
import uuid
import zipfile

import cv2
from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / ".labguard-work" / "roboflow_import"
DEFAULT_DATASET2_OUTPUT = PROJECT_ROOT / ".labguard-work" / "roboflow_import_2"
SOURCE_SPLITS = ("train", "valid", "test")
SPLIT_MAPPING = {"train": "train", "valid": "val", "test": "test"}
SOURCE_CLASS_NAMES = (
    "Safety_Glasses",
    "face_mask",
    "gloves",
    "goggles",
    "lab_coat",
    "person",
)
TARGET_CLASS_NAMES = (
    "person",
    "lab_coat",
    "protective_gloves",
    "safety_goggles",
    "lab_cap",
    "face_mask",
)
CLASS_MAPPING = {0: 3, 1: 5, 2: 2, 3: 3, 4: 1, 5: 0}
PROVISIONAL_SOURCE_CLASSES = frozenset({0, 3})
DATASET2_SOURCE_CLASS_NAMES = (
    "Blue Lab Coat",
    "Full Lab Coat",
    "Glasses",
    "Gloves",
    "Hair Cap",
    "Lab Coat",
    "Mask",
    "No glasses",
    "No gloves",
    "No hair cap",
    "No lab coat",
    "No mask",
    "Shoe Cover",
)
DATASET2_CLASS_MAPPING: dict[int, int | None] = {
    0: 1,
    1: 1,
    2: 3,
    3: 2,
    4: 4,
    5: 1,
    6: 5,
    7: None,
    8: None,
    9: None,
    10: None,
    11: None,
    12: None,
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"^(?P<source>.+)\.rf\.(?P<variant>[0-9a-fA-F]{32})$")
VIDEO_FRAME = re.compile(r"^(?P<video>.+)_mp4-(?P<frame>\d+)_jpg$")
NAMED_SEQUENCE_FRAME = re.compile(
    r"^(?P<sequence>.*?frame)[_-]?(?P<frame>\d+)(?:_jpg)?$",
    re.IGNORECASE,
)
MAX_ARCHIVE_MEMBERS = 100_000
MAX_MEMBER_SIZE = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_SIZE = 5 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000
COPY_CHUNK_SIZE = 1024 * 1024
MAX_ENCODED_IMAGE_SIZE = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000


class DatasetImportError(RuntimeError):
    """Indica que el ZIP no se puede importar de forma segura."""


@dataclass(frozen=True, slots=True)
class ImportProfile:
    """Contrato explicito de clases y descartes para un dataset Roboflow."""

    profile_id: str
    source_class_names: tuple[str, ...]
    class_mapping: dict[int, int | None]
    provisional_source_classes: frozenset[int]
    allowed_duplicate_members: frozenset[str] = frozenset()
    boundary_clip_tolerance: float = 0.0


DEFAULT_IMPORT_PROFILE = ImportProfile(
    profile_id="labguard_dataset_1",
    source_class_names=SOURCE_CLASS_NAMES,
    class_mapping={source: target for source, target in CLASS_MAPPING.items()},
    provisional_source_classes=PROVISIONAL_SOURCE_CLASSES,
)
DATASET2_IMPORT_PROFILE = ImportProfile(
    profile_id="labguard_dataset_2",
    source_class_names=DATASET2_SOURCE_CLASS_NAMES,
    class_mapping=DATASET2_CLASS_MAPPING,
    provisional_source_classes=frozenset({2}),
    allowed_duplicate_members=frozenset({"data.yaml"}),
    boundary_clip_tolerance=2e-5,
)
IMPORT_PROFILES = {
    DEFAULT_IMPORT_PROFILE.profile_id: DEFAULT_IMPORT_PROFILE,
    DATASET2_IMPORT_PROFILE.profile_id: DATASET2_IMPORT_PROFILE,
}


@dataclass(frozen=True, slots=True)
class ConvertedAnnotation:
    """Anotacion convertida con trazabilidad hacia su clase y formato originales."""

    source_class_id: int
    source_class_name: str
    target_class_id: int
    target_class_name: str
    bounding_box: tuple[float, float, float, float]
    original_bounding_box: tuple[float, float, float, float]
    boundary_adjusted: bool
    source_format: str
    polygon_point_count: int
    provisional_merge: bool
    original_line: str
    line_number: int

    def yolo_line(self) -> str:
        coordinates = " ".join(_format_coordinate(value) for value in self.bounding_box)
        return f"{self.target_class_id} {coordinates}"

    def provenance(self, label_member: str) -> dict[str, Any]:
        return {
            "source_label": label_member,
            "line_number": self.line_number,
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "source_format": self.source_format,
            "polygon_point_count": self.polygon_point_count,
            "original_bounding_box": list(self.original_bounding_box),
            "converted_bounding_box": list(self.bounding_box),
            "boundary_adjusted": self.boundary_adjusted,
            "provisional_merge": self.provisional_merge,
            "original_annotation": self.original_line,
        }


@dataclass(frozen=True, slots=True)
class ExcludedAnnotation:
    """Caja valida que el perfil excluye del esquema oficial LabGuard."""

    source_class_id: int
    source_class_name: str
    bounding_box: tuple[float, float, float, float]
    original_bounding_box: tuple[float, float, float, float]
    boundary_adjusted: bool
    source_format: str
    polygon_point_count: int
    original_line: str
    line_number: int

    def provenance(self, label_member: str) -> dict[str, Any]:
        return {
            "source_label": label_member,
            "line_number": self.line_number,
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "source_format": self.source_format,
            "polygon_point_count": self.polygon_point_count,
            "original_bounding_box": list(self.original_bounding_box),
            "converted_bounding_box": list(self.bounding_box),
            "boundary_adjusted": self.boundary_adjusted,
            "original_annotation": self.original_line,
            "exclusion_reason": "source_class_excluded_by_profile",
        }


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    canonical_frame: str
    roboflow_variant: str | None
    video_origin: str | None
    video_frame: int | None
    sequence_origin: str | None
    sequence_frame: int | None


@dataclass(frozen=True, slots=True)
class ImageRecord:
    source_split: str
    target_split: str
    image_member: str
    label_member: str | None
    source_image_path: str
    converted_image_path: str
    converted_label_path: str
    image_sha256: str
    label_sha256: str | None
    width: int | None
    height: int | None
    identity: SourceIdentity
    annotations: tuple[ConvertedAnnotation, ...]
    excluded_annotations: tuple[ExcludedAnnotation, ...]
    label_error_count: int
    verified_empty_label: bool


def _format_coordinate(value: float) -> str:
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _sha256_stream(source: Any) -> str:
    digest = hashlib.sha256()
    source.seek(0)
    for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    path = PurePosixPath(name)
    mode = (info.external_attr >> 16) & 0o170000
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise DatasetImportError(f"Ruta insegura dentro del ZIP: {name!r}")
    if mode and stat.S_ISLNK(mode):
        raise DatasetImportError(f"No se permiten enlaces simbolicos dentro del ZIP: {name}")
    return path


def _validate_archive_members(
    infos: list[zipfile.ZipInfo],
    allowed_duplicate_members: frozenset[str] = frozenset(),
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise DatasetImportError(f"El ZIP supera el limite de {MAX_ARCHIVE_MEMBERS} miembros")
    total_size = 0
    normalised_paths: dict[str, str] = {}
    validated = []
    for info in infos:
        path = _safe_member_path(info)
        normalised = path.as_posix()
        collision_key = normalised.casefold()
        if collision_key in normalised_paths:
            if not (
                normalised in allowed_duplicate_members
                and normalised_paths[collision_key] == info.filename
            ):
                raise DatasetImportError(
                    "Dos miembros del ZIP colisionan al normalizar sus rutas: "
                    f"{normalised_paths[collision_key]!r} y {info.filename!r}"
                )
        else:
            normalised_paths[collision_key] = info.filename
        if not info.is_dir():
            if info.file_size > MAX_MEMBER_SIZE:
                raise DatasetImportError(f"Miembro demasiado grande: {info.filename}")
            total_size += info.file_size
            if total_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
                raise DatasetImportError("El ZIP supera el limite de tamano descomprimido")
            if (
                info.file_size > COPY_CHUNK_SIZE
                and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO
            ):
                raise DatasetImportError(f"Relacion de compresion sospechosa: {info.filename}")
        validated.append((info, path))
    return validated


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with archive.open(info, "r") as source, destination.open("wb") as output:
        while chunk := source.read(COPY_CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if written != info.file_size:
        raise DatasetImportError(f"Tamano descomprimido inesperado para {info.filename}")
    return digest.hexdigest()


def _source_identity(relative_path: PurePosixPath) -> SourceIdentity:
    stem = Path(relative_path.name).stem
    match = ROBOFLOW_SUFFIX.fullmatch(stem)
    canonical = match.group("source") if match else stem
    variant = match.group("variant").lower() if match else None
    video_match = VIDEO_FRAME.fullmatch(canonical)
    sequence_match = NAMED_SEQUENCE_FRAME.fullmatch(canonical)
    parent = relative_path.parent
    canonical_key = (parent / canonical).as_posix() if parent != PurePosixPath(".") else canonical
    video = video_match.group("video") if video_match else None
    video_key = (parent / video).as_posix() if video and parent != PurePosixPath(".") else video
    sequence = sequence_match.group("sequence") if sequence_match and not video_match else None
    sequence_key = (
        (parent / sequence).as_posix()
        if sequence and parent != PurePosixPath(".")
        else sequence
    )
    return SourceIdentity(
        canonical_frame=canonical_key,
        roboflow_variant=variant,
        video_origin=video_key,
        video_frame=int(video_match.group("frame")) if video_match else None,
        sequence_origin=sequence_key,
        sequence_frame=int(sequence_match.group("frame")) if sequence else None,
    )


def _validate_profile(profile: ImportProfile) -> None:
    expected = set(range(len(profile.source_class_names)))
    if set(profile.class_mapping) != expected:
        raise DatasetImportError("El perfil no define exactamente todos los IDs de clase fuente")
    invalid_targets = {
        target
        for target in profile.class_mapping.values()
        if target is not None and target not in range(len(TARGET_CLASS_NAMES))
    }
    if invalid_targets:
        raise DatasetImportError(f"El perfil contiene IDs de destino invalidos: {invalid_targets}")
    if not profile.provisional_source_classes <= expected:
        raise DatasetImportError("El perfil marca clases provisionales desconocidas")
    if not math.isfinite(profile.boundary_clip_tolerance) or not (
        0.0 <= profile.boundary_clip_tolerance <= 1e-3
    ):
        raise DatasetImportError("La tolerancia de recorte del perfil es invalida")


def _parse_class_id(token: str, profile: ImportProfile = DEFAULT_IMPORT_PROFILE) -> int:
    try:
        class_id = int(token)
    except ValueError as error:
        raise ValueError("el ID de clase debe ser un entero") from error
    if str(class_id) != token and token not in {f"+{class_id}", f"-{abs(class_id)}"}:
        raise ValueError("el ID de clase debe usar notacion entera")
    if class_id not in profile.class_mapping:
        raise ValueError(
            f"ID de clase {class_id} fuera del rango 0-{len(profile.source_class_names) - 1}"
        )
    return class_id


def _parse_coordinates(tokens: Iterable[str]) -> tuple[float, ...]:
    try:
        coordinates = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise ValueError("las coordenadas deben ser numericas") from error
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("las coordenadas deben ser finitas")
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError("las coordenadas deben estar normalizadas entre 0 y 1")
    return coordinates


def _validate_box(box: tuple[float, float, float, float]) -> None:
    x_center, y_center, width, height = box
    if width <= 0.0 or height <= 0.0:
        raise ValueError("ancho y alto deben ser mayores que cero")
    if (
        x_center - width / 2 < -1e-6
        or x_center + width / 2 > 1.0 + 1e-6
        or y_center - height / 2 < -1e-6
        or y_center + height / 2 > 1.0 + 1e-6
    ):
        raise ValueError("la caja sale de los limites normalizados")


def _clip_box_to_boundary(
    box: tuple[float, float, float, float],
    tolerance: float,
) -> tuple[tuple[float, float, float, float], bool]:
    """Recorta solo excesos de redondeo permitidos por el perfil."""

    x_center, y_center, width, height = box
    if width <= 0.0 or height <= 0.0:
        raise ValueError("ancho y alto deben ser mayores que cero")
    left = x_center - width / 2
    top = y_center - height / 2
    right = x_center + width / 2
    bottom = y_center + height / 2
    overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
    if overflow == 0.0:
        return box, False
    if tolerance <= 0.0 or overflow > tolerance:
        _validate_box(box)
        return box, False
    clipped_left = max(0.0, left)
    clipped_top = max(0.0, top)
    clipped_right = min(1.0, right)
    clipped_bottom = min(1.0, bottom)
    clipped = (
        (clipped_left + clipped_right) / 2,
        (clipped_top + clipped_bottom) / 2,
        clipped_right - clipped_left,
        clipped_bottom - clipped_top,
    )
    _validate_box(clipped)
    return clipped, True


def _polygon_box(coordinates: tuple[float, ...]) -> tuple[tuple[float, float, float, float], int]:
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError("un poligono requiere al menos tres pares x/y")
    points = tuple(zip(coordinates[::2], coordinates[1::2], strict=True))
    if len(set(points)) < 3:
        raise ValueError("un poligono requiere al menos tres puntos distintos")
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2
    if area <= 1e-12:
        raise ValueError("el poligono no puede tener area cero")

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    minimum_x, maximum_x = min(x_values), max(x_values)
    minimum_y, maximum_y = min(y_values), max(y_values)
    box = (
        (minimum_x + maximum_x) / 2,
        (minimum_y + maximum_y) / 2,
        maximum_x - minimum_x,
        maximum_y - minimum_y,
    )
    _validate_box(box)
    return box, len(points)


def parse_annotation_line(
    line: str,
    *,
    line_number: int = 1,
    profile: ImportProfile = DEFAULT_IMPORT_PROFILE,
) -> ConvertedAnnotation | ExcludedAnnotation:
    """Convierte una linea YOLO de deteccion o segmentacion a una caja LabGuard."""

    fields = line.split()
    if len(fields) < 5:
        raise ValueError("la anotacion tiene menos de cinco campos")
    source_class_id = _parse_class_id(fields[0], profile)
    coordinates = _parse_coordinates(fields[1:])
    if len(fields) == 5:
        original_bounding_box = tuple(coordinates)
        bounding_box, boundary_adjusted = _clip_box_to_boundary(
            original_bounding_box,
            profile.boundary_clip_tolerance,
        )
        source_format = "detection"
        polygon_point_count = 0
    else:
        bounding_box, polygon_point_count = _polygon_box(coordinates)
        original_bounding_box = bounding_box
        boundary_adjusted = False
        source_format = "segmentation"

    target_class_id = profile.class_mapping[source_class_id]
    if target_class_id is None:
        return ExcludedAnnotation(
            source_class_id=source_class_id,
            source_class_name=profile.source_class_names[source_class_id],
            bounding_box=bounding_box,
            original_bounding_box=original_bounding_box,
            boundary_adjusted=boundary_adjusted,
            source_format=source_format,
            polygon_point_count=polygon_point_count,
            original_line=line,
            line_number=line_number,
        )
    return ConvertedAnnotation(
        source_class_id=source_class_id,
        source_class_name=profile.source_class_names[source_class_id],
        target_class_id=target_class_id,
        target_class_name=TARGET_CLASS_NAMES[target_class_id],
        bounding_box=bounding_box,
        original_bounding_box=original_bounding_box,
        boundary_adjusted=boundary_adjusted,
        source_format=source_format,
        polygon_point_count=polygon_point_count,
        provisional_merge=source_class_id in profile.provisional_source_classes,
        original_line=line,
        line_number=line_number,
    )


def _normalise_yaml_names(raw_names: Any) -> tuple[str, ...] | None:
    if isinstance(raw_names, list) and all(isinstance(name, str) for name in raw_names):
        return tuple(raw_names)
    if isinstance(raw_names, dict):
        try:
            class_ids = sorted(int(class_id) for class_id in raw_names)
            if class_ids != list(range(len(raw_names))):
                return None
            return tuple(str(raw_names.get(class_id, raw_names.get(str(class_id)))) for class_id in class_ids)
        except (TypeError, ValueError):
            return None
    return None


def _load_source_metadata(
    source_root: Path,
    profile: ImportProfile = DEFAULT_IMPORT_PROFILE,
) -> dict[str, Any]:
    data_path = source_root / "data.yaml"
    if not data_path.is_file():
        raise DatasetImportError("El ZIP no contiene data.yaml en su raiz")
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DatasetImportError(f"No se pudo leer data.yaml: {error}") from error
    if not isinstance(data, dict):
        raise DatasetImportError("data.yaml debe contener un mapa")
    names = _normalise_yaml_names(data.get("names"))
    if names != profile.source_class_names:
        raise DatasetImportError(
            "Las clases de data.yaml no coinciden con el contrato esperado: "
            + ", ".join(profile.source_class_names)
        )
    number_of_classes = data.get("nc")
    if (
        isinstance(number_of_classes, bool)
        or not isinstance(number_of_classes, int)
        or number_of_classes != len(profile.source_class_names)
    ):
        raise DatasetImportError(f"data.yaml debe declarar nc: {len(profile.source_class_names)}")
    return data


def _add_error(
    errors: list[dict[str, Any]],
    code: str,
    member: str,
    message: str,
    *,
    line_number: int | None = None,
) -> None:
    errors.append(
        {
            "code": code,
            "member": member,
            "line_number": line_number,
            "message": message,
        }
    )


def _class_counts_template(names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(class_id): {"name": name, "total": 0, "by_split": {split: 0 for split in SOURCE_SPLITS}}
        for class_id, name in enumerate(names)
    }


def _group_records(records: list[ImageRecord]) -> dict[str, Any]:
    hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    canonical_frames: dict[str, list[ImageRecord]] = defaultdict(list)
    videos: dict[str, list[ImageRecord]] = defaultdict(list)
    sequences: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        hashes[record.image_sha256].append(record)
        canonical_frames[record.identity.canonical_frame].append(record)
        if record.identity.video_origin is not None:
            videos[record.identity.video_origin].append(record)
        if record.identity.sequence_origin is not None:
            sequences[record.identity.sequence_origin].append(record)

    exact_duplicates = []
    for digest, members in sorted(hashes.items()):
        if len(members) < 2:
            continue
        exact_duplicates.append(
            {
                "image_sha256": digest,
                "count": len(members),
                "source_splits": sorted({member.source_split for member in members}),
                "cross_split": len({member.source_split for member in members}) > 1,
                "conflicting_label_hashes": len(
                    {member.label_sha256 for member in members if member.label_sha256 is not None}
                )
                > 1,
                "members": [member.image_member for member in members],
            }
        )

    augmented_variants = []
    for canonical, members in sorted(canonical_frames.items()):
        variants = {
            member.identity.roboflow_variant
            for member in members
            if member.identity.roboflow_variant is not None
        }
        if len(members) < 2 or len(variants) < 2:
            continue
        augmented_variants.append(
            {
                "canonical_frame": canonical,
                "count": len(members),
                "source_splits": sorted({member.source_split for member in members}),
                "cross_split": len({member.source_split for member in members}) > 1,
                "roboflow_variants": [member.identity.roboflow_variant for member in members],
                "members": [member.image_member for member in members],
            }
        )

    video_origins = []
    for video, members in sorted(videos.items()):
        frames = sorted(
            {member.identity.video_frame for member in members if member.identity.video_frame is not None}
        )
        split_counts = Counter(member.source_split for member in members)
        video_origins.append(
            {
                "video_origin": video,
                "image_count": len(members),
                "unique_frame_count": len(frames),
                "minimum_frame": frames[0] if frames else None,
                "maximum_frame": frames[-1] if frames else None,
                "source_split_counts": {split: split_counts[split] for split in SOURCE_SPLITS},
                "cross_split": sum(count > 0 for count in split_counts.values()) > 1,
                "frames": frames,
            }
        )

    name_sequences = []
    for sequence, members in sorted(sequences.items()):
        frames = sorted(
            {
                member.identity.sequence_frame
                for member in members
                if member.identity.sequence_frame is not None
            }
        )
        split_counts = Counter(member.source_split for member in members)
        name_sequences.append(
            {
                "sequence_origin": sequence,
                "image_count": len(members),
                "unique_frame_count": len(frames),
                "minimum_frame": frames[0] if frames else None,
                "maximum_frame": frames[-1] if frames else None,
                "source_split_counts": {split: split_counts[split] for split in SOURCE_SPLITS},
                "cross_split": sum(count > 0 for count in split_counts.values()) > 1,
                "frames": frames,
            }
        )

    return {
        "exact_duplicate_group_count": len(exact_duplicates),
        "exact_duplicate_image_count": sum(group["count"] for group in exact_duplicates),
        "cross_split_exact_duplicate_group_count": sum(
            group["cross_split"] for group in exact_duplicates
        ),
        "conflicting_label_duplicate_group_count": sum(
            group["conflicting_label_hashes"] for group in exact_duplicates
        ),
        "exact_duplicates": exact_duplicates,
        "augmented_variant_group_count": len(augmented_variants),
        "augmented_variant_image_count": sum(group["count"] for group in augmented_variants),
        "cross_split_augmented_variant_group_count": sum(
            group["cross_split"] for group in augmented_variants
        ),
        "augmented_variants": augmented_variants,
        "video_origin_group_count": len(video_origins),
        "cross_split_video_group_count": sum(group["cross_split"] for group in video_origins),
        "video_origins": video_origins,
        "name_sequence_group_count": len(name_sequences),
        "cross_split_name_sequence_group_count": sum(
            group["cross_split"] for group in name_sequences
        ),
        "name_sequences": name_sequences,
    }


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    first_edges = (
        first_x - first_w / 2,
        first_y - first_h / 2,
        first_x + first_w / 2,
        first_y + first_h / 2,
    )
    second_edges = (
        second_x - second_w / 2,
        second_y - second_h / 2,
        second_x + second_w / 2,
        second_y + second_h / 2,
    )
    intersection_width = max(0.0, min(first_edges[2], second_edges[2]) - max(first_edges[0], second_edges[0]))
    intersection_height = max(0.0, min(first_edges[3], second_edges[3]) - max(first_edges[1], second_edges[1]))
    intersection = intersection_width * intersection_height
    union = first_w * first_h + second_w * second_h - intersection
    return intersection / union if union > 0 else 0.0


def _coat_overlap_candidates(
    image_member: str,
    label_member: str,
    annotations: list[ConvertedAnnotation],
) -> list[dict[str, Any]]:
    coats = [annotation for annotation in annotations if annotation.target_class_id == 1]
    candidates = []
    for index, first in enumerate(coats):
        for second in coats[index + 1 :]:
            if first.source_class_id == second.source_class_id:
                continue
            overlap = _bbox_iou(first.bounding_box, second.bounding_box)
            if overlap < 0.8:
                continue
            candidates.append(
                {
                    "source_image": image_member,
                    "source_label": label_member,
                    "iou": overlap,
                    "classification": "near_duplicate" if overlap >= 0.98 else "high_overlap",
                    "first": {
                        "line_number": first.line_number,
                        "source_class_id": first.source_class_id,
                        "source_class_name": first.source_class_name,
                        "bounding_box": list(first.bounding_box),
                    },
                    "second": {
                        "line_number": second.line_number,
                        "source_class_id": second.source_class_id,
                        "source_class_name": second.source_class_name,
                        "bounding_box": list(second.bounding_box),
                    },
                    "action": "manual_review_required_no_automatic_deletion",
                }
            )
    return candidates


def _compare_reference_staging(
    records: list[ImageRecord],
    reference_staging: Path | None,
) -> dict[str, Any]:
    if reference_staging is None:
        return {
            "reference_staging": None,
            "exact_duplicate_count": 0,
            "exact_duplicates": [],
            "canonical_name_match_count": 0,
            "canonical_name_matches": [],
        }
    provenance_path = reference_staging / "provenance" / "images.jsonl"
    if not provenance_path.is_file():
        raise DatasetImportError(f"No existe la procedencia del staging de referencia: {provenance_path}")
    reference_records = []
    for line_number, line in enumerate(provenance_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetImportError(
                f"Procedencia de referencia invalida en linea {line_number}"
            ) from error
        reference_records.append(item)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_canonical_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reference_records:
        by_hash[str(item.get("image_sha256"))].append(item)
        canonical = item.get("canonical_frame")
        if canonical:
            by_canonical_name[PurePosixPath(str(canonical)).name.casefold()].append(item)
    exact = []
    names = []
    for record in records:
        if record.image_sha256 in by_hash:
            exact.append(
                {
                    "source_image": record.image_member,
                    "image_sha256": record.image_sha256,
                    "reference_images": [item["source_image"] for item in by_hash[record.image_sha256]],
                }
            )
        canonical_key = PurePosixPath(record.identity.canonical_frame).name.casefold()
        if canonical_key in by_canonical_name:
            names.append(
                {
                    "source_image": record.image_member,
                    "canonical_frame": record.identity.canonical_frame,
                    "reference_images": [
                        item["source_image"] for item in by_canonical_name[canonical_key]
                    ],
                    "note": "Coincidencia nominal; requiere comparacion visual.",
                }
            )
    return {
        "reference_staging": str(reference_staging),
        "exact_duplicate_count": len(exact),
        "exact_duplicates": exact,
        "canonical_name_match_count": len(names),
        "canonical_name_matches": names,
    }


def _annotation_color(annotation: ConvertedAnnotation) -> tuple[int, int, int]:
    colors = (
        (220, 170, 55),
        (220, 180, 40),
        (70, 210, 120),
        (245, 180, 60),
        (180, 100, 240),
        (70, 170, 245),
    )
    return colors[annotation.target_class_id]


def _select_review_records(
    records: list[ImageRecord],
    samples_per_class: int,
    profile: ImportProfile = DEFAULT_IMPORT_PROFILE,
) -> list[ImageRecord]:
    selected: dict[str, ImageRecord] = {}
    for source_class_id in range(len(profile.source_class_names)):
        for split in SOURCE_SPLITS:
            candidates = [
                record
                for record in records
                if record.source_split == split
                and any(
                    annotation.source_class_id == source_class_id
                    for annotation in record.annotations + record.excluded_annotations
                )
            ]
            for record in candidates[:samples_per_class]:
                selected[record.image_member] = record
    for record in records:
        if any(annotation.source_format == "segmentation" for annotation in record.annotations):
            selected[record.image_member] = record
    for split in SOURCE_SPLITS:
        empty = next(
            (
                record
                for record in records
                if record.source_split == split
                and record.verified_empty_label
                and record.width is not None
            ),
            None,
        )
        if empty is not None:
            selected[empty.image_member] = empty
    return [selected[key] for key in sorted(selected)]


def _create_visual_review(
    staging_root: Path,
    records: list[ImageRecord],
    samples_per_class: int,
    errors: list[dict[str, Any]],
    profile: ImportProfile = DEFAULT_IMPORT_PROFILE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    review_root = staging_root / "review"
    images_root = review_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    selected = _select_review_records(records, samples_per_class, profile)

    for index, record in enumerate(selected, start=1):
        image_path = staging_root / record.converted_image_path
        try:
            image = cv2.imread(str(image_path))
        except cv2.error as error:
            _add_error(errors, "review_decode_error", record.image_member, str(error))
            continue
        if image is None:
            _add_error(errors, "review_decode_error", record.image_member, "OpenCV no pudo abrir la muestra")
            continue
        height, width = image.shape[:2]
        labels = []
        for annotation in record.annotations:
            x_center, y_center, box_width, box_height = annotation.bounding_box
            x1 = max(0, min(width - 1, round((x_center - box_width / 2) * width)))
            y1 = max(0, min(height - 1, round((y_center - box_height / 2) * height)))
            x2 = max(0, min(width - 1, round((x_center + box_width / 2) * width)))
            y2 = max(0, min(height - 1, round((y_center + box_height / 2) * height)))
            color = _annotation_color(annotation)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            qualifiers = []
            if annotation.provisional_merge:
                qualifiers.append("PROVISIONAL")
            if annotation.source_format == "segmentation":
                qualifiers.append("POLY->BOX")
            suffix = f" [{' | '.join(qualifiers)}]" if qualifiers else ""
            label = f"{annotation.target_class_name} <- {annotation.source_class_name}{suffix}"
            labels.append(label)
            text_y = max(14, y1 - 5)
            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        for annotation in record.excluded_annotations:
            x_center, y_center, box_width, box_height = annotation.bounding_box
            x1 = max(0, min(width - 1, round((x_center - box_width / 2) * width)))
            y1 = max(0, min(height - 1, round((y_center - box_height / 2) * height)))
            x2 = max(0, min(width - 1, round((x_center + box_width / 2) * width)))
            y2 = max(0, min(height - 1, round((y_center + box_height / 2) * height)))
            color = (80, 80, 240)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"EXCLUIDA <- {annotation.source_class_name}"
            labels.append(label)
            cv2.putText(
                image,
                label,
                (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(record.image_member).name)
        output_name = f"{index:03d}_{record.source_split}_{safe_name}"
        output_path = images_root / output_name
        if not cv2.imwrite(str(output_path), image):
            _add_error(errors, "review_write_error", record.image_member, "OpenCV no pudo guardar la muestra")
            continue
        manifest.append(
            {
                "source_image": record.image_member,
                "source_split": record.source_split,
                "review_image": f"images/{output_name}",
                "canonical_frame": record.identity.canonical_frame,
                "video_origin": record.identity.video_origin,
                "video_frame": record.identity.video_frame,
                "labels": labels,
            }
        )

    (review_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = []
    for item in manifest:
        labels_html = "<br>".join(html.escape(label) for label in item["labels"]) or "Sin objetos"
        rows.append(
            "<article>"
            f"<h2>{html.escape(item['source_split'])}: {html.escape(item['canonical_frame'])}</h2>"
            f"<img src=\"{html.escape(item['review_image'])}\" alt=\"Muestra anotada\">"
            f"<p>{labels_html}</p>"
            f"<code>{html.escape(item['source_image'])}</code>"
            "</article>"
        )
    provisional_names = ", ".join(
        profile.source_class_names[class_id]
        for class_id in sorted(profile.provisional_source_classes)
    )
    index_html = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Revision Roboflow LabGuard</title>
<style>body{font-family:system-ui;max-width:1100px;margin:auto;padding:2rem;background:#101820;color:#eef}
article{border:1px solid #456;padding:1rem;margin:1rem 0;border-radius:.5rem}img{max-width:100%;height:auto}
code{overflow-wrap:anywhere;color:#9ed}</style></head><body>
<h1>Revision visual de conversion Roboflow</h1>
<p>PROVISIONAL requiere confirmacion humana. EXCLUIDA se conserva solo en la auditoria y no pasa a YOLO.</p>
<p>Clases provisionales de origen: """ + html.escape(provisional_names or "ninguna") + ".</p>\n" + "\n".join(rows) + "\n</body></html>\n"
    (review_root / "index.html").write_text(index_html, encoding="utf-8")
    (review_root / "README.md").write_text(
        "# Revision visual\n\n"
        "Desde la raiz del proyecto:\n\n"
        "```bash\n"
        f"python -m http.server 8000 --directory \"{output_path / 'review'}\"\n"
        "```\n\n"
        "Abre http://localhost:8000 y revisa especialmente PROVISIONAL, EXCLUIDA y POLY->BOX.\n",
        encoding="utf-8",
    )
    return {
        "sample_count": len(manifest),
        "index": "review/index.html",
        "manifest": "review/manifest.json",
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_reports(staging_root: Path, report: dict[str, Any]) -> None:
    reports_root = staging_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "import_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (reports_root / "class_counts.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["namespace", "class_id", "class_name", "total", *SOURCE_SPLITS])
        for namespace in ("source", "target"):
            counts = report[f"{namespace}_class_counts"]
            for class_id, item in counts.items():
                writer.writerow(
                    [
                        namespace,
                        class_id,
                        item["name"],
                        item["total"],
                        *(item["by_split"][split] for split in SOURCE_SPLITS),
                    ]
                )

    with (reports_root / "errors.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("code", "member", "line_number", "message"))
        writer.writeheader()
        writer.writerows(report["errors"])

    source_rows = "\n".join(
        f"| {class_id} | `{item['name']}` | {item['total']} |"
        for class_id, item in report["source_class_counts"].items()
    )
    target_rows = "\n".join(
        f"| {class_id} | `{item['name']}` | {item['total']} |"
        for class_id, item in report["target_class_counts"].items()
    )
    excluded_rows = "\n".join(
        f"| {class_id} | `{item['name']}` | {item['total']} |"
        for class_id, item in report.get("excluded_class_counts", {}).items()
        if item["total"]
    ) or "| - | Ninguna | 0 |"
    error_rows = "\n".join(
        f"- `{error['code']}` {error['member']}"
        + (f":{error['line_number']}" if error["line_number"] else "")
        + f": {error['message']}"
        for error in report["errors"][:30]
    ) or "- Ninguno."
    blockers = "\n".join(f"- {blocker}" for blocker in report["known_blockers"])
    summary = report["summary"]
    groups = report["groups"]
    markdown = f"""# Reporte de importacion Roboflow

## Fuente

- ZIP: `{report['source']['zip_path']}`
- SHA-256: `{report['source']['zip_sha256']}`
- Licencia declarada: `{report['source']['declared_license']}`
- Splits conservados: `train -> train`, `valid -> val`, `test -> test`

## Resumen

- Imagenes: {summary['images']}
- Etiquetas: {summary['labels']}
- Etiquetas vacias validas: {summary['empty_labels']}
- Anotaciones convertidas: {summary['annotations']}
- Anotaciones excluidas y conservadas en auditoria: {summary.get('excluded_annotations', 0)}
- Cajas ajustadas por redondeo en el borde: {summary.get('boundary_adjustments', 0)}
- Cajas originales: {report['formats']['detection']}
- Poligonos convertidos a caja: {report['formats']['segmentation']}
- Errores: {len(report['errors'])}
- Listo para entrenamiento: **no**

## Clases originales

| ID | Clase | Objetos |
|---:|---|---:|
{source_rows}

## Clases LabGuard convertidas

| ID | Clase | Objetos |
|---:|---|---:|
{target_rows}

## Clases excluidas del esquema LabGuard

| ID original | Clase | Cajas excluidas |
|---:|---|---:|
{excluded_rows}

## Grupos

- Duplicados exactos: {groups['exact_duplicate_group_count']} grupos.
- Variantes Roboflow: {groups['augmented_variant_group_count']} grupos.
- Videos de origen: {groups['video_origin_group_count']} grupos.
- Secuencias inferidas por nombre: {groups.get('name_sequence_group_count', 0)} grupos.
- Duplicados exactos entre splits: {groups['cross_split_exact_duplicate_group_count']} grupos.
- Variantes Roboflow entre splits: {groups['cross_split_augmented_variant_group_count']} grupos.
- Videos presentes en mas de un split: {groups['cross_split_video_group_count']}.
- Secuencias nominales presentes en mas de un split: {groups.get('cross_split_name_sequence_group_count', 0)}.
- Candidatos de batas duplicadas o muy superpuestas: {len(report.get('coat_overlap_candidates', []))}.
- Coincidencias exactas con el primer staging: {groups.get('external_comparison', {}).get('exact_duplicate_count', 0)}.
- Coincidencias nominales con el primer staging: {groups.get('external_comparison', {}).get('canonical_name_match_count', 0)}.

## Bloqueos antes de entrenar

{blockers}

## Errores

{error_rows}

## Revision visual

Abre `review/index.html`. `PROVISIONAL` requiere confirmacion humana; `EXCLUIDA` no se incorpora a las etiquetas convertidas.
"""
    (reports_root / "import_report.md").write_text(markdown, encoding="utf-8")


def _validate_output_location(output_dir: Path) -> None:
    forbidden = (PROJECT_ROOT / "data" / "labguard").resolve()
    resolved = output_dir.resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise DatasetImportError("El staging no puede escribirse dentro de data/labguard")
    if resolved.exists():
        raise DatasetImportError(f"La carpeta de salida ya existe: {resolved}")


def import_roboflow_archive(
    zip_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    samples_per_class: int = 2,
    profile: ImportProfile = DEFAULT_IMPORT_PROFILE,
    reference_staging: str | Path | None = None,
) -> dict[str, Any]:
    """Importa, convierte y documenta el ZIP sin modificarlo ni redistribuir splits."""

    archive_path = Path(zip_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    reference_path = (
        Path(reference_staging).expanduser().resolve() if reference_staging is not None else None
    )
    if not archive_path.is_file():
        raise DatasetImportError(f"No existe el ZIP: {archive_path}")
    if samples_per_class < 0:
        raise DatasetImportError("samples_per_class no puede ser negativo")
    _validate_profile(profile)
    _validate_output_location(output_path)

    archive_size = 0
    archive_hash = ""
    previous_umask = os.umask(0o077)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staging_root = output_path.parent / f".{output_path.name}.building-{uuid.uuid4().hex}"
        staging_root.mkdir(mode=0o700)
    except Exception:
        os.umask(previous_umask)
        raise
    errors: list[dict[str, Any]] = []
    records: list[ImageRecord] = []
    annotation_provenance: list[dict[str, Any]] = []
    excluded_provenance: list[dict[str, Any]] = []
    boundary_adjustments: list[dict[str, Any]] = []
    coat_overlap_candidates: list[dict[str, Any]] = []
    provisional_review: list[dict[str, Any]] = []
    member_provenance: list[dict[str, Any]] = []
    source_counts = _class_counts_template(profile.source_class_names)
    target_counts = _class_counts_template(TARGET_CLASS_NAMES)
    excluded_counts = _class_counts_template(profile.source_class_names)
    format_counts = Counter()
    empty_labels = 0

    try:
        source_root = staging_root / "source"
        member_hashes: dict[str, str] = {}
        with archive_path.open("rb") as archive_source:
            initial_stat = os.fstat(archive_source.fileno())
            archive_size = initial_stat.st_size
            archive_hash = _sha256_stream(archive_source)
            with zipfile.ZipFile(archive_source) as archive:
                validated_members = _validate_archive_members(
                    archive.infolist(), profile.allowed_duplicate_members
                )
                for info, relative in validated_members:
                    normalised_name = relative.as_posix()
                    source_path = source_root.joinpath(*relative.parts)
                    if info.is_dir():
                        source_path.mkdir(parents=True, exist_ok=True)
                        member_provenance.append(
                            {
                                "zip_member": info.filename,
                                "normalised_member": normalised_name,
                                "member_type": "directory",
                                "preserved_path": str(source_path.relative_to(staging_root)),
                                "size_bytes": 0,
                                "compressed_size_bytes": info.compress_size,
                                "sha256": None,
                            }
                        )
                        continue
                    try:
                        digest = _extract_member(archive, info, source_path)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                        raise DatasetImportError(f"No se pudo leer {info.filename}: {error}") from error
                    previous_digest = member_hashes.get(normalised_name)
                    if previous_digest is not None and previous_digest != digest:
                        raise DatasetImportError(
                            f"Miembros duplicados con contenido diferente: {normalised_name}"
                        )
                    member_hashes[normalised_name] = digest
                    member_provenance.append(
                        {
                            "zip_member": info.filename,
                            "normalised_member": normalised_name,
                            "member_type": "file",
                            "preserved_path": str(source_path.relative_to(staging_root)),
                            "size_bytes": info.file_size,
                            "compressed_size_bytes": info.compress_size,
                            "sha256": digest,
                        }
                    )
            final_stat = os.fstat(archive_source.fileno())
            final_hash = _sha256_stream(archive_source)
            if (
                (initial_stat.st_dev, initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns)
                != (final_stat.st_dev, final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns)
                or final_hash != archive_hash
            ):
                raise DatasetImportError("El ZIP cambio mientras se realizaba la importacion")

        metadata = _load_source_metadata(source_root, profile)
        roboflow_metadata = metadata.get("roboflow", {})
        if not isinstance(roboflow_metadata, dict):
            roboflow_metadata = {}

        all_members = set(member_hashes)
        all_label_members = {
            member
            for member in all_members
            if any(member.startswith(f"{split}/labels/") for split in SOURCE_SPLITS)
            and Path(member).suffix.lower() == ".txt"
        }
        expected_labels: set[str] = set()
        converted_image_destinations: set[str] = set()
        converted_label_destinations: set[str] = set()

        for source_split in SOURCE_SPLITS:
            image_prefix = f"{source_split}/images/"
            image_members = sorted(
                member
                for member in all_members
                if member.startswith(image_prefix) and Path(member).suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_members:
                _add_error(errors, "empty_split", source_split, "el split no contiene imagenes")

            for image_member in image_members:
                image_relative = PurePosixPath(image_member).relative_to(image_prefix)
                label_relative = image_relative.with_suffix(".txt")
                label_member = f"{source_split}/labels/{label_relative.as_posix()}"
                expected_labels.add(label_member)
                target_split = SPLIT_MAPPING[source_split]
                identity = _source_identity(image_relative)
                source_image = source_root.joinpath(*PurePosixPath(image_member).parts)
                converted_image = (
                    staging_root / "converted" / "images" / target_split / Path(*image_relative.parts)
                )
                image_destination_key = str(converted_image.relative_to(staging_root)).casefold()
                if image_destination_key in converted_image_destinations:
                    raise DatasetImportError(f"Colision de imagen convertida: {image_member}")
                converted_image_destinations.add(image_destination_key)
                converted_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_image, converted_image)

                width = height = None
                if source_image.stat().st_size > MAX_ENCODED_IMAGE_SIZE:
                    _add_error(
                        errors,
                        "invalid_image",
                        image_member,
                        "la imagen codificada supera el limite permitido",
                    )
                else:
                    try:
                        with Image.open(source_image) as image:
                            width, height = image.size
                            if width * height > MAX_IMAGE_PIXELS:
                                raise ValueError("la imagen supera el limite de pixeles permitido")
                            image.verify()
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        width = height = None
                        _add_error(errors, "invalid_image", image_member, str(error))

                converted_label = (
                    staging_root / "converted" / "labels" / target_split / Path(*label_relative.parts)
                )
                label_destination_key = str(converted_label.relative_to(staging_root)).casefold()
                if label_destination_key in converted_label_destinations:
                    raise DatasetImportError(f"Colision de etiqueta convertida: {label_member}")
                converted_label_destinations.add(label_destination_key)
                converted_label.parent.mkdir(parents=True, exist_ok=True)
                annotations: list[ConvertedAnnotation] = []
                excluded_annotations: list[ExcludedAnnotation] = []
                label_error_count = 0
                label_sha256 = None
                verified_empty_label = False
                if label_member not in all_members:
                    _add_error(errors, "missing_label", label_member, f"falta la etiqueta de {image_member}")
                    label_error_count += 1
                else:
                    label_path = source_root.joinpath(*PurePosixPath(label_member).parts)
                    label_content = label_path.read_bytes()
                    label_sha256 = member_hashes[label_member]
                    if not label_content:
                        empty_labels += 1
                    try:
                        lines = label_content.decode("utf-8").splitlines()
                    except UnicodeError as error:
                        _add_error(errors, "label_encoding", label_member, str(error))
                        lines = []
                        label_error_count += 1
                    if not label_error_count and not any(line.strip() for line in lines):
                        verified_empty_label = True
                        if label_content:
                            empty_labels += 1
                    seen_lines: set[str] = set()
                    for line_number, line in enumerate(lines, start=1):
                        if not line.strip():
                            continue
                        normalised_line = " ".join(line.split())
                        if normalised_line in seen_lines:
                            _add_error(
                                errors,
                                "duplicate_annotation",
                                label_member,
                                "anotacion duplicada",
                                line_number=line_number,
                            )
                            label_error_count += 1
                            continue
                        seen_lines.add(normalised_line)
                        try:
                            annotation = parse_annotation_line(
                                line,
                                line_number=line_number,
                                profile=profile,
                            )
                        except ValueError as error:
                            _add_error(
                                errors,
                                "invalid_annotation",
                                label_member,
                                str(error),
                                line_number=line_number,
                            )
                            label_error_count += 1
                            continue
                        format_counts[annotation.source_format] += 1
                        source_item = source_counts[str(annotation.source_class_id)]
                        source_item["total"] += 1
                        source_item["by_split"][source_split] += 1
                        if isinstance(annotation, ExcludedAnnotation):
                            excluded_annotations.append(annotation)
                            excluded_item = excluded_counts[str(annotation.source_class_id)]
                            excluded_item["total"] += 1
                            excluded_item["by_split"][source_split] += 1
                            provenance = annotation.provenance(label_member)
                            provenance.update(
                                {
                                    "source_image": image_member,
                                    "source_split": source_split,
                                    "target_split": target_split,
                                    "canonical_frame": identity.canonical_frame,
                                    "roboflow_variant": identity.roboflow_variant,
                                    "video_origin": identity.video_origin,
                                    "video_frame": identity.video_frame,
                                    "sequence_origin": identity.sequence_origin,
                                    "sequence_frame": identity.sequence_frame,
                                }
                            )
                            excluded_provenance.append(provenance)
                            if annotation.boundary_adjusted:
                                boundary_adjustments.append(provenance)
                            continue
                        annotations.append(annotation)
                        target_item = target_counts[str(annotation.target_class_id)]
                        target_item["total"] += 1
                        target_item["by_split"][source_split] += 1
                        provenance = annotation.provenance(label_member)
                        provenance.update(
                            {
                                "source_image": image_member,
                                "source_split": source_split,
                                "target_split": target_split,
                                "canonical_frame": identity.canonical_frame,
                                "roboflow_variant": identity.roboflow_variant,
                                "video_origin": identity.video_origin,
                                "video_frame": identity.video_frame,
                                "sequence_origin": identity.sequence_origin,
                                "sequence_frame": identity.sequence_frame,
                            }
                        )
                        annotation_provenance.append(provenance)
                        if annotation.boundary_adjusted:
                            boundary_adjustments.append(provenance)
                        if annotation.provisional_merge:
                            provisional_review.append(provenance)

                coat_overlap_candidates.extend(
                    _coat_overlap_candidates(image_member, label_member, annotations)
                )

                converted_label.write_text(
                    "".join(f"{annotation.yolo_line()}\n" for annotation in annotations),
                    encoding="utf-8",
                )
                records.append(
                    ImageRecord(
                        source_split=source_split,
                        target_split=target_split,
                        image_member=image_member,
                        label_member=label_member if label_member in all_members else None,
                        source_image_path=str(source_image.relative_to(staging_root)),
                        converted_image_path=str(converted_image.relative_to(staging_root)),
                        converted_label_path=str(converted_label.relative_to(staging_root)),
                        image_sha256=member_hashes[image_member],
                        label_sha256=label_sha256,
                        width=width,
                        height=height,
                        identity=identity,
                        annotations=tuple(annotations),
                        excluded_annotations=tuple(excluded_annotations),
                        label_error_count=label_error_count,
                        verified_empty_label=verified_empty_label,
                    )
                )

        for orphan in sorted(all_label_members - expected_labels):
            _add_error(errors, "orphan_label", orphan, "etiqueta sin imagen correspondiente")

        groups = _group_records(records)
        groups["external_comparison"] = _compare_reference_staging(records, reference_path)
        image_provenance = [
            {
                "source_split": record.source_split,
                "target_split": record.target_split,
                "source_image": record.image_member,
                "source_label": record.label_member,
                "preserved_image": record.source_image_path,
                "converted_image": record.converted_image_path,
                "converted_label": record.converted_label_path,
                "image_sha256": record.image_sha256,
                "label_sha256": record.label_sha256,
                "width": record.width,
                "height": record.height,
                "canonical_frame": record.identity.canonical_frame,
                "roboflow_variant": record.identity.roboflow_variant,
                "video_origin": record.identity.video_origin,
                "video_frame": record.identity.video_frame,
                "sequence_origin": record.identity.sequence_origin,
                "sequence_frame": record.identity.sequence_frame,
                "annotation_count": len(record.annotations),
                "excluded_annotation_count": len(record.excluded_annotations),
                "label_error_count": record.label_error_count,
            }
            for record in records
        ]
        _write_jsonl(staging_root / "provenance" / "images.jsonl", image_provenance)
        _write_jsonl(staging_root / "provenance" / "annotations.jsonl", annotation_provenance)
        _write_jsonl(
            staging_root / "provenance" / "excluded_annotations.jsonl",
            excluded_provenance,
        )
        _write_jsonl(
            staging_root / "provenance" / "boundary_adjustments.jsonl",
            boundary_adjustments,
        )
        _write_jsonl(
            staging_root / "provenance" / "coat_overlap_candidates.jsonl",
            coat_overlap_candidates,
        )
        _write_jsonl(staging_root / "provenance" / "archive_members.jsonl", member_provenance)
        _write_jsonl(staging_root / "review" / "provisional_safety_goggles.jsonl", provisional_review)

        visual_review = _create_visual_review(
            staging_root,
            records,
            samples_per_class,
            errors,
            profile,
            output_path,
        )
        provisional_names = ", ".join(
            profile.source_class_names[class_id]
            for class_id in sorted(profile.provisional_source_classes)
        )
        known_blockers = []
        if provisional_review:
            known_blockers.append(
                f"{provisional_names} se convirtio provisionalmente y requiere revision visual."
            )
        missing_targets = [
            item["name"] for item in target_counts.values() if item["total"] == 0
        ]
        if missing_targets:
            known_blockers.append(
                "Clases LabGuard sin anotaciones: " + ", ".join(missing_targets) + "."
            )
        if excluded_provenance:
            known_blockers.append(
                "Las clases negativas o fuera del esquema se excluyeron de YOLO y permanecen auditadas por separado."
            )
        if coat_overlap_candidates:
            known_blockers.append(
                "Hay cajas de distintas clases de batas duplicadas o muy superpuestas; no se eliminaron y requieren revision."
            )
        if groups["cross_split_exact_duplicate_group_count"]:
            known_blockers.append("Hay imagenes exactamente duplicadas presentes en splits diferentes.")
        if groups["cross_split_augmented_variant_group_count"]:
            known_blockers.append("Hay variantes Roboflow del mismo origen presentes en splits diferentes.")
        if groups["cross_split_video_group_count"]:
            known_blockers.append(
                "Hay videos de origen presentes en varios splits; existe riesgo de fuga temporal aunque no se redistribuyeron datos."
            )
        if groups["cross_split_name_sequence_group_count"]:
            known_blockers.append(
                "Hay secuencias inferidas por nombre presentes en varios splits; deben agruparse antes de una futura redistribucion."
            )
        if groups["conflicting_label_duplicate_group_count"]:
            known_blockers.append("Hay imagenes identicas con archivos de etiqueta diferentes.")
        if groups["external_comparison"]["exact_duplicate_count"]:
            known_blockers.append("Hay imagenes exactamente duplicadas respecto al primer staging.")
        if groups["external_comparison"]["canonical_name_match_count"]:
            known_blockers.append(
                "Hay coincidencias nominales respecto al primer staging que requieren comparacion visual."
            )
        if errors:
            known_blockers.append("El reporte contiene errores que deben resolverse antes de cualquier entrenamiento.")

        image_split_counts = Counter(record.source_split for record in records)
        label_split_counts = Counter(
            record.source_split for record in records if record.label_member is not None
        )
        empty_split_counts = Counter(
            record.source_split for record in records if record.verified_empty_label
        )
        polygon_conversions = [
            provenance
            for provenance in annotation_provenance
            if provenance["source_format"] == "segmentation"
        ]

        report = {
            "schema_version": 1,
            "profile_id": profile.profile_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "zip_path": str(archive_path),
                "zip_size_bytes": archive_size,
                "zip_sha256": archive_hash,
                "archive_member_count": len(member_provenance),
                "roboflow": roboflow_metadata,
                "declared_license": str(roboflow_metadata.get("license", "no declarada")),
                "original_files_preserved_under": "source/",
                "duplicate_archive_members": {
                    name: count
                    for name, count in Counter(
                        item["normalised_member"] for item in member_provenance
                    ).items()
                    if count > 1
                },
            },
            "split_mapping": SPLIT_MAPPING,
            "class_mapping": [
                {
                    "source_class_id": source_id,
                    "source_class_name": profile.source_class_names[source_id],
                    "target_class_id": target_id,
                    "target_class_name": (
                        TARGET_CLASS_NAMES[target_id] if target_id is not None else None
                    ),
                    "provisional_merge": source_id in profile.provisional_source_classes,
                    "excluded": target_id is None,
                }
                for source_id, target_id in profile.class_mapping.items()
            ],
            "summary": {
                "images": len(records),
                "labels": sum(record.label_member is not None for record in records),
                "empty_labels": empty_labels,
                "annotations": len(annotation_provenance),
                "source_annotations_valid": len(annotation_provenance)
                + len(excluded_provenance),
                "excluded_annotations": len(excluded_provenance),
                "boundary_adjustments": len(boundary_adjustments),
                "provisional_annotations": len(provisional_review),
                "by_source_split": {
                    split: {
                        "images": image_split_counts[split],
                        "labels": label_split_counts[split],
                        "empty_labels": empty_split_counts[split],
                    }
                    for split in SOURCE_SPLITS
                },
                "ready_for_training": False,
            },
            "formats": {
                "detection": format_counts["detection"],
                "segmentation": format_counts["segmentation"],
                "polygon_to_bounding_box": format_counts["segmentation"],
                "polygon_conversion_records": polygon_conversions,
            },
            "source_class_counts": source_counts,
            "target_class_counts": target_counts,
            "excluded_class_counts": excluded_counts,
            "groups": groups,
            "coat_overlap_candidates": coat_overlap_candidates,
            "visual_review": visual_review,
            "errors": errors,
            "known_blockers": known_blockers,
        }
        _write_reports(staging_root, report)
        os.replace(staging_root, output_path)
        return report
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    finally:
        os.umask(previous_umask)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="ZIP Roboflow que se lee sin modificar.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile",
        choices=tuple(IMPORT_PROFILES),
        default=DEFAULT_IMPORT_PROFILE.profile_id,
        help="Contrato de clases fuente y conversion.",
    )
    parser.add_argument(
        "--reference-staging",
        type=Path,
        help="Staging previo de solo lectura para buscar coincidencias exactas o nominales.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Muestras por clase original y split para el indice visual.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = import_roboflow_archive(
            args.zip,
            args.output,
            samples_per_class=args.samples_per_class,
            profile=IMPORT_PROFILES[args.profile],
            reference_staging=args.reference_staging,
        )
    except (DatasetImportError, OSError, zipfile.BadZipFile) as error:
        print(f"Error: {error}")
        return 2

    summary = report["summary"]
    print(f"Importacion completada en: {args.output.expanduser().resolve()}")
    print(
        f"Imagenes={summary['images']} | anotaciones={summary['annotations']} | "
        f"errores={len(report['errors'])} | poligonos convertidos={report['formats']['segmentation']}"
    )
    print("Reporte: reports/import_report.md")
    print("Revision visual: review/index.html")
    print("Estado: no listo para entrenamiento; revise known_blockers en el reporte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
