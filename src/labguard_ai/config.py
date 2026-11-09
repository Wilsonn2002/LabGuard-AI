"""Carga y validacion de la configuracion de LabGuard AI."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labguard_ai.core import PpeType


class ModelConfigurationError(ValueError):
    """Indica que la configuracion del modelo no es valida."""


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """Parametros necesarios para construir el detector local."""

    model_path: Path
    repository: str
    revision: str
    class_mapping: Mapping[str, PpeType]
    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    person_model_path: Path | None = None
    person_confidence_threshold: float = 0.25
    person_iou_threshold: float = 0.45
    maximum_people: int = 50


def _read_yaml(config_path: Path) -> Any:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Falta PyYAML. Instala las dependencias del proyecto para leer la configuracion."
        ) from error

    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelConfigurationError(f"No se pudo leer {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ModelConfigurationError(f"El YAML de {config_path} no es valido") from error


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ModelConfigurationError(f"Falta la seccion '{name}' en la configuracion")
    return section


def load_model_configuration(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ModelConfiguration:
    """Lee model.yaml y resuelve la ruta del checkpoint desde el proyecto."""
    path = Path(config_path).expanduser().resolve()
    data = _read_yaml(path)
    if not isinstance(data, Mapping):
        raise ModelConfigurationError("La configuracion debe contener un mapa YAML")

    model = _section(data, "model")
    inference = _section(data, "inference")
    raw_mapping = _section(data, "class_mapping")
    person_detection = data.get("person_detection", {})
    if not isinstance(person_detection, Mapping):
        raise ModelConfigurationError("person_detection debe ser un mapa YAML")

    local_path = model.get("local_path")
    if not isinstance(local_path, str) or not local_path.strip():
        raise ModelConfigurationError("model.local_path debe indicar un archivo .pt")

    root = Path(project_root).resolve() if project_root else path.parent.parent
    resolved_model_path = Path(local_path).expanduser()
    if not resolved_model_path.is_absolute():
        resolved_model_path = root / resolved_model_path

    try:
        class_mapping = {
            str(source_class): PpeType(target_class)
            for source_class, target_class in raw_mapping.items()
        }
        device = str(inference.get("device", "cpu")).lower()
        image_size = int(inference["image_size"])
        confidence_threshold = float(inference["confidence_threshold"])
        iou_threshold = float(inference["iou_threshold"])
        person_confidence_threshold = float(
            person_detection.get("confidence_threshold", 0.25)
        )
        person_iou_threshold = float(person_detection.get("iou_threshold", 0.45))
        maximum_people = int(person_detection.get("maximum_people", 50))
    except (KeyError, TypeError, ValueError) as error:
        raise ModelConfigurationError("Los parametros de inferencia o clases no son validos") from error

    if device != "cpu":
        raise ModelConfigurationError("La primera interfaz de LabGuard AI solo permite device: cpu")
    if not class_mapping:
        raise ModelConfigurationError("class_mapping no puede estar vacio")
    if image_size <= 0:
        raise ModelConfigurationError("inference.image_size debe ser mayor que cero")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ModelConfigurationError("inference.confidence_threshold debe estar entre 0 y 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ModelConfigurationError("inference.iou_threshold debe estar entre 0 y 1")
    if not 0.0 <= person_confidence_threshold <= 1.0:
        raise ModelConfigurationError(
            "person_detection.confidence_threshold debe estar entre 0 y 1"
        )
    if not 0.0 <= person_iou_threshold <= 1.0:
        raise ModelConfigurationError("person_detection.iou_threshold debe estar entre 0 y 1")
    if maximum_people <= 0:
        raise ModelConfigurationError("person_detection.maximum_people debe ser positivo")

    person_local_path = person_detection.get("local_path", "yolov8n.pt")
    if not isinstance(person_local_path, str) or not person_local_path.strip():
        raise ModelConfigurationError("person_detection.local_path debe indicar un archivo .pt")
    resolved_person_model_path = Path(person_local_path).expanduser()
    if not resolved_person_model_path.is_absolute():
        resolved_person_model_path = root / resolved_person_model_path

    return ModelConfiguration(
        model_path=resolved_model_path.resolve(),
        repository=str(model.get("repository", "")),
        revision=str(model.get("revision", "")),
        class_mapping=class_mapping,
        device=device,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        person_model_path=resolved_person_model_path.resolve(),
        person_confidence_threshold=person_confidence_threshold,
        person_iou_threshold=person_iou_threshold,
        maximum_people=maximum_people,
    )
