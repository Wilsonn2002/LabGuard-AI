"""Adaptador futuro para YOLOv8 y pesos alojados en Hugging Face."""

from dataclasses import dataclass

from labguard_ai.core import Detection


@dataclass(frozen=True, slots=True)
class YoloModelReference:
    """Ubicacion reproducible de un modelo en Hugging Face Hub."""

    repository: str
    filename: str


class YoloPpeDetector:
    """Limite de integracion con Ultralytics; no carga modelos al importarse."""

    def __init__(self, model: YoloModelReference) -> None:
        self.model = model

    def detect(self, image: object) -> list[Detection]:
        """Ejecuta inferencia cuando se implemente la integracion del modelo."""
        raise NotImplementedError("La integracion con YOLOv8 se implementara en una fase posterior")
