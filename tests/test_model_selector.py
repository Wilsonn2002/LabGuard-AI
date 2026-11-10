from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from labguard_ai.core.model_selector import (
    get_model_choices,
    load_selected_configuration,
)


class TestModelSelector(unittest.TestCase):
    def test_definitivo_disabled_when_yaml_missing_or_invalid(self):
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            def fake_load_model_configuration(config_path, *, project_root=None):
                # Consideramos que solo falla el definitivo
                if "model_definitivo.yaml" in str(config_path):
                    raise RuntimeError("No se pudo leer el YAML")
                # provisional "OK"
                return object()

            with patch(
                "labguard_ai.core.model_selector.load_model_configuration",
                side_effect=fake_load_model_configuration,
            ):
                provisional, definitivo = get_model_choices(project_root=project_root)

            self.assertTrue(provisional.available)
            self.assertFalse(definitivo.available)
            self.assertIsNotNone(definitivo.error_message)
            self.assertIn("No se pudo leer el YAML", definitivo.error_message)

    def test_load_selected_configuration_raises_when_definitivo_unavailable(self):
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            def fake_load_model_configuration(config_path, *, project_root=None):
                raise RuntimeError("Definitivo no válido")

            with patch(
                "labguard_ai.core.model_selector.load_model_configuration",
                side_effect=fake_load_model_configuration,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    load_selected_configuration(
                        mode="definitivo",
                        project_root=project_root,
                    )

            self.assertIn("Definitivo no disponible", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

