"""Adaptador YOLOv8 para detectar EPP en imagenes de OpenCV."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from labguard_ai.core import Detection, PpeType

ClassKey = str | int
ModelLoader = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class YoloModelReference:
    """Ubicacion reproducible de un modelo en Hugging Face Hub."""

    repository: str
    filename: str


class YoloPpeDetector:
    """Ejecuta un modelo YOLOv8 local y traduce sus clases a EPP."""

    def __init__(
        self,
        model_path: str | Path,
        class_mapping: Mapping[ClassKey, PpeType | str] | None = None,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 512,
        model_loader: ModelLoader | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".pt":
            raise ValueError("El modelo YOLO debe ser un archivo .pt")
        if not path.is_file():
            raise FileNotFoundError(f"No se encontro el modelo YOLO: {path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold debe estar entre 0 y 1")
        if image_size <= 0:
            raise ValueError("image_size debe ser mayor que cero")

        default_mapping = {item.value: item for item in PpeType}
        source_mapping = default_mapping if class_mapping is None else class_mapping

        self.model_path = path
        self.class_mapping = {
            source_class: PpeType(target_class)
            for source_class, target_class in source_mapping.items()
        }
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self._model_loader = model_loader or self._load_ultralytics_model
        self._model: Any | None = None

    @staticmethod
    def _load_ultralytics_model(model_path: str) -> Any:
        """Importa Ultralytics solo cuando se necesita cargar el modelo."""
        from ultralytics import YOLO

        return YOLO(model_path)

    @staticmethod
    def _prepare_image(image: object) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser una matriz NumPy de OpenCV")
        if image.size == 0:
            raise ValueError("La imagen no puede estar vacia")
        if image.dtype != np.uint8:
            raise TypeError("La imagen debe usar el tipo uint8")

        image = np.ascontiguousarray(image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3:
            raise ValueError("La imagen debe tener dos o tres dimensiones")
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] != 3:
            raise ValueError("La imagen debe tener uno, tres o cuatro canales")
        return image

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        """Convierte tensores de PyTorch o valores simulados sin importar torch."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(str(self.model_path))
        return self._model

    def _source_class_name(self, result: Any, class_id: int) -> str | None:
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self._get_model(), "names", None)
        if isinstance(names, Mapping):
            name = names.get(class_id, names.get(str(class_id)))
            return None if name is None else str(name)
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return None

    def _mapped_class(self, result: Any, class_id: int) -> PpeType | None:
        mapped = self.class_mapping.get(class_id)
        if mapped is not None:
            return mapped

        source_name = self._source_class_name(result, class_id)
        if source_name is None:
            return None
        return self.class_mapping.get(source_name)

    def detect(self, image: object) -> list[Detection]:
        """Detecta y convierte las cajas YOLO correspondientes a EPP conocido."""
        prepared_image = self._prepare_image(image)
        results = self._get_model().predict(
            source=prepared_image,
            device=self.device,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            class_ids = self._as_numpy(boxes.cls).reshape(-1)
            confidences = self._as_numpy(boxes.conf).reshape(-1)
            coordinates = self._as_numpy(boxes.xyxy).reshape(-1, 4)

            for raw_class_id, raw_confidence, raw_box in zip(
                class_ids, confidences, coordinates, strict=True
            ):
                confidence = float(raw_confidence)
                if confidence < self.confidence_threshold:
                    continue

                item = self._mapped_class(result, int(raw_class_id))
                if item is None:
                    continue

                bounding_box = tuple(int(round(float(value))) for value in raw_box)
                detections.append(
                    Detection(
                        item=item,
                        confidence=confidence,
                        bounding_box=bounding_box,
                    )
                )

        return detections
