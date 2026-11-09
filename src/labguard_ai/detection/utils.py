"""Conversiones compartidas por los adaptadores YOLO."""

from typing import Any

import cv2
import numpy as np


def prepare_opencv_image(image: object) -> np.ndarray:
    """Valida y normaliza una imagen NumPy al formato BGR de OpenCV."""
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


def as_numpy(value: Any) -> np.ndarray:
    """Convierte tensores de PyTorch o valores simulados sin importar torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
