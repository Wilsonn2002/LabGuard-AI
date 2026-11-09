"""Contrato para implementar detectores YOLO u otros modelos."""

from collections.abc import Sequence
from typing import Protocol

from labguard_ai.core import Detection


class PpeDetector(Protocol):
    """Interfaz que desacopla el modelo del resto de la aplicacion."""

    def detect(self, image: object) -> Sequence[Detection]:
        """Detecta EPP en una imagen ya decodificada."""
        ...
