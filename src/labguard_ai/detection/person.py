"""Deteccion de personas con una clase person de un modelo YOLO."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from labguard_ai.core import PersonDetection
from labguard_ai.detection.utils import as_numpy, prepare_opencv_image

ModelLoader = Callable[[str], Any]


class PersonDetector(Protocol):
    """Contrato para detectar personas dentro de una imagen."""

    def detect(self, image: object) -> Sequence[PersonDetection]:
        """Devuelve personas con identificadores temporales por fotografia."""
        ...


class YoloPersonDetector:
    """Usa un checkpoint local y limita la inferencia a su clase person."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 512,
        maximum_detections: int = 50,
        person_class: str | int = "person",
        model_loader: ModelLoader | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".pt":
            raise ValueError("El detector de personas debe ser un archivo .pt")
        if not path.is_file():
            raise FileNotFoundError(f"No se encontro el modelo de personas: {path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold debe estar entre 0 y 1")
        if image_size <= 0 or maximum_detections <= 0:
            raise ValueError("image_size y maximum_detections deben ser positivos")
        if isinstance(person_class, str) and not person_class.strip():
            raise ValueError("person_class no puede estar vacia")
        if isinstance(person_class, int) and person_class < 0:
            raise ValueError("person_class no puede ser negativa")

        self.model_path = path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.maximum_detections = maximum_detections
        self.person_class = person_class
        self._model_loader = model_loader or self._load_ultralytics_model
        self._model: Any | None = None

    @staticmethod
    def _load_ultralytics_model(model_path: str) -> Any:
        from ultralytics import YOLO

        return YOLO(model_path)

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(str(self.model_path))
        return self._model

    def _person_class_id(self, model: Any) -> int:
        if isinstance(self.person_class, int):
            return self.person_class

        names = getattr(model, "names", None)
        if isinstance(names, Mapping):
            for class_id, class_name in names.items():
                if str(class_name) == self.person_class:
                    return int(class_id)
        elif isinstance(names, (list, tuple)):
            for class_id, class_name in enumerate(names):
                if str(class_name) == self.person_class:
                    return class_id
        raise ValueError(
            f"El modelo {self.model_path} no contiene la clase '{self.person_class}'"
        )

    def detect(self, image: object) -> list[PersonDetection]:
        prepared_image = prepare_opencv_image(image)
        model = self._get_model()
        person_class_id = self._person_class_id(model)
        results = model.predict(
            source=prepared_image,
            device=self.device,
            classes=[person_class_id],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=self.maximum_detections,
            verbose=False,
        )

        raw_people: list[tuple[float, tuple[int, int, int, int]]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            confidences = as_numpy(boxes.conf).reshape(-1)
            coordinates = as_numpy(boxes.xyxy).reshape(-1, 4)
            for raw_confidence, raw_box in zip(confidences, coordinates, strict=True):
                confidence = float(raw_confidence)
                if confidence < self.confidence_threshold:
                    continue
                bounding_box = tuple(int(round(float(value))) for value in raw_box)
                raw_people.append((confidence, bounding_box))

        raw_people.sort(key=lambda item: (item[1][0], item[1][1], -item[0]))
        return [
            PersonDetection(
                person_id=f"Persona {index}",
                confidence=confidence,
                bounding_box=bounding_box,
            )
            for index, (confidence, bounding_box) in enumerate(raw_people, start=1)
        ]
