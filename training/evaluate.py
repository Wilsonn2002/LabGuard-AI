"""Evalua un checkpoint YOLO y exporta metricas globales y por clase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .validate_dataset import CLASS_NAMES, ordered_class_names, validate_dataset
except ImportError:  # Permite ejecutar: python training/evaluate.py
    from validate_dataset import CLASS_NAMES, ordered_class_names, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(values)


def per_class_metrics(box_metrics: Any) -> list[dict[str, float | int | str | None]]:
    """Convierte las matrices de Ultralytics en metricas asociadas al ID correcto."""

    class_ids = [int(value) for value in _as_list(box_metrics.ap_class_index)]
    precision = _as_list(box_metrics.p)
    recall = _as_list(box_metrics.r)
    average_precision = _as_list(box_metrics.all_ap)
    positions = {class_id: index for index, class_id in enumerate(class_ids)}
    rows: list[dict[str, float | int | str | None]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        position = positions.get(class_id)
        if position is None:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": None,
                    "recall": None,
                    "map50": None,
                    "map50_95": None,
                }
            )
            continue
        ap_values = _as_list(average_precision[position])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[position]),
                "recall": float(recall[position]),
                "map50": float(ap_values[0]),
                "map50_95": float(sum(ap_values) / len(ap_values)),
            }
        )
    return rows


def model_class_names(model: Any) -> tuple[str, ...] | None:
    """Normaliza los nombres del checkpoint conservando el orden de sus IDs."""

    return ordered_class_names(model.names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint best.pt o last.pt local.")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_dataset(args.data)
    print(report.format())
    if not report.valid:
        return 2

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Error: no existe el checkpoint local: {model_path}")
        return 2

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "detect":
        print(f"Error: el checkpoint es de tarea {model.task!r}, no de deteccion.")
        return 2
    checkpoint_names = model_class_names(model)
    if checkpoint_names != CLASS_NAMES:
        print(
            "Error: las clases del checkpoint no coinciden con el contrato LabGuard: "
            f"{', '.join(CLASS_NAMES)}"
        )
        return 2
    metrics = model.val(
        data=str(args.data.expanduser().resolve()),
        split=args.split,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=args.plots,
    )
    result = {
        "model": str(model_path),
        "split": args.split,
        "overall": {key: float(value) for key, value in metrics.results_dict.items()},
        "classes": per_class_metrics(metrics.box),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("\nMetricas por clase:")
    for row in result["classes"]:
        if row["precision"] is None:
            print(f"  {row['class_name']}: sin ejemplos evaluables")
        else:
            print(
                f"  {row['class_name']}: P={row['precision']:.4f} R={row['recall']:.4f} "
                f"mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}"
            )
    print(f"Reporte JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
