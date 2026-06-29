from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def train_yolo_barcode(
    data: str | Path = "data/barcode_dataset/data.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str | Path = "models/training_runs",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"status": "error", "message": f"Ultralytics indisponible: {exc}"}

    data_path = _absolute(data)
    project_path = _absolute(project)
    if not data_path.exists():
        return {"status": "error", "message": f"Dataset YAML introuvable: {data_path}"}

    project_path.mkdir(parents=True, exist_ok=True)
    try:
        yolo = YOLO(model)
        results = yolo.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project_path),
            name="barcode_yolov8",
            exist_ok=True,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Echec entrainement YOLO: {exc}"}

    save_dir = Path(getattr(results, "save_dir", project_path / "barcode_yolov8"))
    best_source = save_dir / "weights" / "best.pt"
    final_model = PROJECT_ROOT / "models" / "barcode_yolov8.pt"
    final_model.parent.mkdir(parents=True, exist_ok=True)
    if best_source.exists():
        shutil.copy2(best_source, final_model)

    metrics: dict[str, Any] = {
        "status": "success",
        "data": str(data_path),
        "base_model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "training_run": str(save_dir),
        "best_model": str(final_model) if final_model.exists() else "",
    }
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        with results_csv.open(encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if rows:
            metrics["last_epoch_metrics"] = rows[-1]

    metrics_path = save_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainer YOLOv8 pour localiser les codes-barres.")
    parser.add_argument("--data", default="data/barcode_dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="models/training_runs")
    args = parser.parse_args()
    print(json.dumps(train_yolo_barcode(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
