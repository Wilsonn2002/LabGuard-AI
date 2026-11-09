"""Prueba de humo de la interfaz Streamlit sin cargar una imagen."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTest(unittest.TestCase):
    def test_app_starts_with_expected_inference_controls(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path).run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 1)
        sliders = {slider.label: slider for slider in app.slider}
        self.assertEqual(sliders["Umbral de confianza"].min, 0.10)
        self.assertEqual(sliders["Umbral de confianza"].max, 0.60)
        self.assertEqual(sliders["Tamano de inferencia"].min, 512)
        self.assertEqual(sliders["Tamano de inferencia"].max, 640)
        self.assertEqual(len(app.toggle), 1)
        self.assertEqual(app.toggle[0].label, "Mostrar advertencias visuales")
        self.assertFalse(app.toggle[0].value)


if __name__ == "__main__":
    unittest.main()
