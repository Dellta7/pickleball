from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 detect model for pickleball (ball/player/paddle)")
    parser.add_argument("--data", type=Path, default=Path("training/pickleball.yaml"), help="Dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model or checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8, help="For 8GB VRAM: try 4/8; reduce if OOM")
    parser.add_argument("--device", type=str, default="0", help="'0' for first GPU, or 'cpu'")
    parser.add_argument("--workers", type=int, default=4)
    # Ultralytics tự tạo subfolder theo task (detect/pose/...) dưới project.
    # Để tránh bị lồng kiểu runs/detect/runs/detect/... hãy để project = runs.
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", type=str, default="pickleball")
    args = parser.parse_args()

    data_path = args.data
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        patience=30,
        cos_lr=True,
        amp=True,
    )


if __name__ == "__main__":
    main()
