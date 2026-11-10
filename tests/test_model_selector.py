from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from labguard_ai.core.model_selector import (
    get_model_choices,
    load_selected_configuration,
)


def test_definitivo_disabled_when_yaml_missing_or_invalid(tmp_path: Path):
    # Simulamos que provisional carga OK y definitivo falla.
    project_root = tmp_path

    def fake_load_model_configuration(config_path, *, project_root=None):
        if "model_definitivo.yaml" in str(config_path):
            raise RuntimeError("No se pudo leer el YAML")
        return object()

    with patch(
        "labguard_ai.core.model_selector.load_model_configuration",
        side_effect=fake_load_model_configuration,
    ):
        provisional, definitivo = get_model_choices(project_root=project_root)

    assert provisional.available is True
    assert definitivo.available is False
    assert "No se pudo leer" in (definitivo.error_message or "")


def test_load_selected_configuration_raises_when_definitivo_unavailable(tmp_path: Path):
    project_root = tmp_path

    def fake_load_model_configuration(config_path, *, project_root=None):
        raise RuntimeError("Definitivo no válido")

    with patch(
        "labguard_ai.core.model_selector.load_model_configuration",
        side_effect=fake_load_model_configuration,
    ):
        with pytest.raises(RuntimeError) as exc:
            load_selected_configuration(
                mode="definitivo",
                project_root=project_root,
            )
    assert "Definitivo no disponible" in str(exc.value)
