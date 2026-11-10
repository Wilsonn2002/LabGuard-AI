from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from labguard_ai.config import ModelConfiguration, load_model_configuration


Mode = Literal["provisional", "definitivo"]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    mode: Mode
    config_path: Path
    available: bool
    error_message: str | None


DEFAULT_PROVISIONAL_PATHNAME = "configs/model.yaml"
DEFAULT_DEFINITIVO_PATHNAME = "configs/model_definitivo.yaml"


def _try_load_config(config_path: Path, *, project_root: Path) -> tuple[bool, str | None]:
    try:
        # Si carga bien, lo consideramos disponible.
        load_model_configuration(config_path, project_root=project_root)
        return True, None
    except Exception as e:  # noqa: BLE001 - UI-friendly fallback info
        return False, str(e)


def get_model_choices(
    *,
    project_root: str | Path,
    provisional_path: str | Path = DEFAULT_PROVISIONAL_PATHNAME,
    definitivo_path: str | Path = DEFAULT_DEFINITIVO_PATHNAME,
) -> tuple[ModelChoice, ModelChoice]:
    root = Path(project_root).resolve()
    provisional_config_path = (root / provisional_path).resolve()
    definitivo_config_path = (root / definitivo_path).resolve()

    provisional_available, provisional_error = _try_load_config(
        provisional_config_path, project_root=root
    )

    definitivo_available, definitivo_error = _try_load_config(
        definitivo_config_path, project_root=root
    )

    return (
        ModelChoice(
            mode="provisional",
            config_path=provisional_config_path,
            available=provisional_available,
            error_message=provisional_error,
        ),
        ModelChoice(
            mode="definitivo",
            config_path=definitivo_config_path,
            available=definitivo_available,
            error_message=definitivo_error,
        ),
    )


def load_selected_configuration(
    *,
    mode: Mode,
    project_root: str | Path,
    provisional_path: str | Path = DEFAULT_PROVISIONAL_PATHNAME,
    definitivo_path: str | Path = DEFAULT_DEFINITIVO_PATHNAME,
) -> ModelConfiguration:
    root = Path(project_root).resolve()

    prov_path = (root / provisional_path).resolve()
    def_path = (root / definitivo_path).resolve()

    config_path = def_path if mode == "definitivo" else prov_path

    # A: si no está disponible, levantamos para que la UI lo trate como "no disponible"
    # (no hacemos fallback silencioso).
    if mode == "definitivo":
        available, err = _try_load_config(def_path, project_root=root)
        if not available:
            raise RuntimeError(f"Definitivo no disponible: {err}")

    return load_model_configuration(config_path, project_root=root)
