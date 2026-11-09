"""Deteccion de EPP mediante modelos de vision."""

from labguard_ai.detection.base import PpeDetector
from labguard_ai.detection.person import PersonDetector, YoloPersonDetector
from labguard_ai.detection.yolo import YoloModelReference, YoloPpeDetector

__all__ = [
    "PersonDetector",
    "PpeDetector",
    "YoloModelReference",
    "YoloPersonDetector",
    "YoloPpeDetector",
]
