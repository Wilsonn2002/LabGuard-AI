"""Valida la estructura y las anotaciones de un dataset de deteccion YOLO."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import yaml


CLASS_NAMES = (
    "person",
    "lab_coat",
    "protective_gloves",
    "safety_goggles",
    "lab_cap",
    "face_mask",
)
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ValidationIssue:
    """Problema encontrado en un archivo o en la configuracion."""

    path: Path
    message: str
    line: int | None = None

    def __str__(self) -> str:
        location = str(self.path)
        if self.line is not None:
            location = f"{location}:{self.line}"
        return f"{location}: {self.message}"


@dataclass
class DatasetValidationReport:
    """Resultado agregado de la validacion del dataset."""

    issues: list[ValidationIssue] = field(default_factory=list)
    image_counts: Counter[str] = field(default_factory=Counter)
    annotation_counts: Counter[str] = field(default_factory=Counter)
    class_counts: Counter[str] = field(default_factory=Counter)
    split_class_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {split: Counter() for split in SPLITS}
    )

    @property
    def valid(self) -> bool:
        return not self.issues

    def format(self) -> str:
        lines = ["Dataset YOLO valido." if self.valid else "Dataset YOLO invalido."]
        for split in SPLITS:
            lines.append(
                f"  {split}: {self.image_counts[split]} imagenes, "
                f"{self.annotation_counts[split]} objetos"
            )
        lines.append("  Objetos por clase:")
        lines.extend(f"    {name}: {self.class_counts[name]}" for name in CLASS_NAMES)
        if self.issues:
            lines.append("  Errores:")
            lines.extend(f"    - {issue}" for issue in self.issues)
        return "\n".join(lines)


def _read_configuration(config_path: Path, report: DatasetValidationReport) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.issues.append(ValidationIssue(config_path, "el archivo de configuracion no existe"))
        return None
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        report.issues.append(ValidationIssue(config_path, f"no se pudo leer el YAML: {error}"))
        return None

    if not isinstance(data, dict):
        report.issues.append(ValidationIssue(config_path, "el YAML debe contener un mapa"))
        return None
    return data


def ordered_class_names(names: Any) -> tuple[str, ...] | None:
    """Normaliza una lista o mapa de nombres con IDs contiguos."""

    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return tuple(names)
    if isinstance(names, dict):
        try:
            ordered_ids = sorted(int(class_id) for class_id in names)
            if ordered_ids != list(range(len(names))):
                return None
            return tuple(str(names.get(class_id, names.get(str(class_id)))) for class_id in ordered_ids)
        except (TypeError, ValueError):
            return None
    return None


def _configured_names(data: dict[str, Any]) -> tuple[str, ...] | None:
    return ordered_class_names(data.get("names"))


def _labels_directory(images_directory: Path) -> Path | None:
    parts = list(images_directory.parts)
    try:
        images_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return None
    parts[images_index] = "labels"
    return Path(*parts)


def _validate_annotation(
    label_path: Path,
    report: DatasetValidationReport,
    split: str,
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        report.issues.append(ValidationIssue(label_path, f"no se pudo leer: {error}"))
        return

    seen: set[tuple[int, float, float, float, float]] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            report.issues.append(
                ValidationIssue(label_path, "se esperaban 5 campos: clase x_centro y_centro ancho alto", line_number)
            )
            continue

        try:
            class_value = float(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError:
            report.issues.append(ValidationIssue(label_path, "todos los campos deben ser numericos", line_number))
            continue

        if not class_value.is_integer():
            report.issues.append(ValidationIssue(label_path, "el ID de clase debe ser entero", line_number))
            continue
        class_id = int(class_value)
        if class_id not in range(len(CLASS_NAMES)):
            report.issues.append(
                ValidationIssue(label_path, f"ID de clase {class_id} fuera del rango 0-{len(CLASS_NAMES) - 1}", line_number)
            )
            continue
        if not all(math.isfinite(value) for value in coordinates):
            report.issues.append(ValidationIssue(label_path, "las coordenadas deben ser finitas", line_number))
            continue

        x_center, y_center, width, height = coordinates
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            report.issues.append(ValidationIssue(label_path, "el centro debe estar entre 0 y 1", line_number))
            continue
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            report.issues.append(ValidationIssue(label_path, "ancho y alto deben estar entre 0 (exclusivo) y 1", line_number))
            continue
        if (
            x_center - width / 2 < -1e-6
            or x_center + width / 2 > 1.0 + 1e-6
            or y_center - height / 2 < -1e-6
            or y_center + height / 2 > 1.0 + 1e-6
        ):
            report.issues.append(ValidationIssue(label_path, "la caja sale de los limites de la imagen", line_number))
            continue

        annotation = (class_id, *coordinates)
        if annotation in seen:
            report.issues.append(ValidationIssue(label_path, "anotacion duplicada", line_number))
            continue
        seen.add(annotation)
        report.annotation_counts[split] += 1
        report.class_counts[CLASS_NAMES[class_id]] += 1
        report.split_class_counts[split][CLASS_NAMES[class_id]] += 1


def _validate_split(
    root: Path,
    split: str,
    images_value: str,
    report: DatasetValidationReport,
    check_images: bool,
) -> None:
    images_directory = root / images_value
    labels_directory = _labels_directory(images_directory)
    if labels_directory is None:
        report.issues.append(
            ValidationIssue(images_directory, "la ruta de imagenes debe incluir un directorio llamado 'images'")
        )
        return
    if not images_directory.is_dir():
        report.issues.append(ValidationIssue(images_directory, f"no existe el directorio de imagenes de {split}"))
        return
    if not labels_directory.is_dir():
        report.issues.append(ValidationIssue(labels_directory, f"no existe el directorio de etiquetas de {split}"))
        return

    images = sorted(
        path for path in images_directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    report.image_counts[split] = len(images)
    if not images:
        report.issues.append(ValidationIssue(images_directory, f"el conjunto {split} no contiene imagenes compatibles"))

    expected_labels: set[Path] = set()
    for image_path in images:
        relative_path = image_path.relative_to(images_directory)
        label_path = labels_directory / relative_path.with_suffix(".txt")
        expected_labels.add(label_path)
        if check_images:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format.lower() if image.format else ""
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                report.issues.append(ValidationIssue(image_path, f"Pillow no puede verificar la imagen: {error}"))
            else:
                supported_formats = {extension.removeprefix(".") for extension in IMAGE_EXTENSIONS} | {"jpeg2000"}
                if image_format not in supported_formats:
                    report.issues.append(
                        ValidationIssue(image_path, f"el formato interno {image_format!r} no es compatible con YOLO")
                    )
                if height <= 9 or width <= 9:
                    report.issues.append(
                        ValidationIssue(image_path, "la imagen debe medir mas de 9 pixeles en cada dimension")
                    )
        if not label_path.is_file():
            report.issues.append(ValidationIssue(label_path, f"falta la etiqueta para {image_path.name}"))
            continue
        _validate_annotation(label_path, report, split)

    for label_path in labels_directory.rglob("*.txt"):
        if label_path not in expected_labels:
            report.issues.append(ValidationIssue(label_path, "etiqueta huerfana sin imagen correspondiente"))


def validate_dataset(config_path: str | Path, *, check_images: bool = True) -> DatasetValidationReport:
    """Valida configuracion, estructura, imagenes y cajas YOLO normalizadas."""

    config_path = Path(config_path).expanduser().resolve()
    report = DatasetValidationReport()
    data = _read_configuration(config_path, report)
    if data is None:
        return report

    names = _configured_names(data)
    if names != CLASS_NAMES:
        report.issues.append(
            ValidationIssue(
                config_path,
                f"names debe declarar exactamente, y en este orden: {', '.join(CLASS_NAMES)}",
            )
        )

    root_value = data.get("path")
    if root_value is None:
        root = config_path.parent
    elif isinstance(root_value, str) and root_value.strip():
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            report.issues.append(
                ValidationIssue(config_path, "path debe ser absoluto; omitalo para usar rutas relativas al YAML")
            )
            return report
    else:
        report.issues.append(ValidationIssue(config_path, "path debe ser una ruta no vacia cuando se declara"))
        return report

    for split in SPLITS:
        images_value = data.get(split)
        if not isinstance(images_value, str) or not images_value.strip():
            report.issues.append(ValidationIssue(config_path, f"falta una ruta valida para {split}"))
            continue
        if Path(images_value).is_absolute():
            report.issues.append(ValidationIssue(config_path, f"la ruta de {split} debe ser relativa a path"))
            continue
        _validate_split(root, split, images_value, report, check_images)

    for split in SPLITS:
        missing_classes = [name for name in CLASS_NAMES if report.split_class_counts[split][name] == 0]
        if missing_classes:
            report.issues.append(
                ValidationIssue(
                    config_path,
                    f"{split} no contiene anotaciones validas para: {', '.join(missing_classes)}",
                )
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset.yaml"))
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="No intenta decodificar las imagenes; conserva las demas validaciones.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_dataset(args.data, check_images=not args.skip_image_check)
    print(report.format())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
