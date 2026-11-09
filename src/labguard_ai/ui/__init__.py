"""Interfaz de usuario de LabGuard AI."""

from labguard_ai.ui.images import (
    annotate_image,
    annotate_people,
    build_face_mask_warnings,
    count_detections,
    decode_uploaded_image,
    draw_warning_triangle,
    summarize_result,
)

__all__ = [
    "annotate_image",
    "annotate_people",
    "build_face_mask_warnings",
    "count_detections",
    "decode_uploaded_image",
    "draw_warning_triangle",
    "summarize_result",
]
