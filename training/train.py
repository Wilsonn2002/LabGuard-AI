"""Ajusta YOLOv8n para detectar personas y EPP de laboratorio."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .validate_dataset import CLASS_NAMES, ordered_class_names, validate_dataset
except ImportError:  # Permite ejecutar: python training/train.py
    from validate_dataset import CLASS_NAMES, ordered_class_names, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset.yaml"))
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="Dispositivo Ultralytics, por ejemplo 0 o cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=0, help="Cantidad de capas iniciales que se congelan.")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--name", default="labguard_yolov8n")
    parser.add_argument("--resume", action="store_true", help="Reanuda el checkpoint indicado en --model.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Confirma que se iniciara entrenamiento. Sin esta opcion solo valida y muestra el plan.",
    )
    return parser


def _positive(value: int, option: str) -> bool:
    if value <= 0:
        print(f"Error: {option} debe ser mayor que cero.")
        return False
    return True


def _nonnegative(value: int, option: str) -> bool:
    if value < 0:
        print(f"Error: {option} no puede ser negativo.")
        return False
    return True


def main() -> int:
    args = build_parser().parse_args()
    report = validate_dataset(args.data)
    print(report.format())
    if not report.valid:
        return 2

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Error: no existe el peso base local: {model_path}")
        return 2
    if not all(
        (
            _positive(args.epochs, "--epochs"),
            _nonnegative(args.patience, "--patience"),
            _positive(args.image_size, "--image-size"),
            _positive(args.batch, "--batch"),
            _nonnegative(args.workers, "--workers"),
        )
    ):
        return 2
    if args.freeze < 0:
        print("Error: --freeze no puede ser negativo.")
        return 2

    print(
        "Plan: "
        f"modelo={model_path}, epochs={args.epochs}, imgsz={args.image_size}, "
        f"batch={args.batch}, device={args.device}, salida={args.project / args.name}"
    )
    if not args.execute:
        print("Entrenamiento no iniciado. Agregue --execute unicamente en el entorno de entrenamiento.")
        return 0

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "detect":
        print(f"Error: el peso base es de tarea {model.task!r}, no de deteccion.")
        return 2
    if args.resume and ordered_class_names(model.names) != CLASS_NAMES:
        print("Error: --resume requiere un checkpoint con las seis clases LabGuard en el orden esperado.")
        return 2
    parameters = {
        "data": str(args.data.expanduser().resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.image_size,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "pretrained": True,
        "resume": args.resume,
        "plots": True,
    }
    if args.freeze:
        parameters["freeze"] = args.freeze
    model.train(**parameters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
