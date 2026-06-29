from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "output" / "evaluation"


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _metric_value(box_metrics: Any, name: str, default: float = 0.0) -> float:
    try:
        value = getattr(box_metrics, name)
        return float(value.item() if hasattr(value, "item") else value)
    except Exception:
        return default


def evaluate_yolo_model(
    model_path: str | Path = "models/barcode_yolov8.pt",
    data: str | Path = "data/barcode_dataset/data.yaml",
    split: str = "val",
    imgsz: int = 640,
) -> dict[str, Any]:
    output_dir = EVALUATION_DIR
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    weights = _absolute(model_path)
    dataset = _absolute(data)

    base = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "avg_inference_time_ms": 0.0,
        "model_path": str(weights),
        "dataset": str(dataset),
        "status": "error",
        "message": "",
        "charts": {},
    }
    if not weights.exists():
        base["message"] = f"Modele YOLO introuvable: {weights}"
        _save_metrics(base, output_dir)
        return base
    if not dataset.exists():
        base["message"] = f"Dataset YAML introuvable: {dataset}"
        _save_metrics(base, output_dir)
        return base

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        results = model.val(
            data=str(dataset),
            split=split,
            imgsz=imgsz,
            plots=True,
            project=str(output_dir / "runs"),
            name="yolo_validation",
            exist_ok=True,
            verbose=False,
        )
    except Exception as exc:
        base["message"] = f"Echec evaluation YOLO: {exc}"
        _save_metrics(base, output_dir)
        return base

    box = getattr(results, "box", None)
    precision = _metric_value(box, "mp")
    recall = _metric_value(box, "mr")
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    speed = getattr(results, "speed", {}) or {}
    save_dir = Path(getattr(results, "save_dir", output_dir / "runs" / "yolo_validation"))

    chart_names = {
        "confusion_matrix": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
        "pr_curve": ["BoxPR_curve.png", "PR_curve.png"],
        "f1_curve": ["BoxF1_curve.png", "F1_curve.png"],
    }
    charts: dict[str, str] = {}
    for key, candidates in chart_names.items():
        for candidate in candidates:
            source = save_dir / candidate
            if source.exists():
                target = charts_dir / candidate
                shutil.copy2(source, target)
                charts[key] = str(target)
                break

    metrics = {
        **base,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1_score, 6),
        "map50": round(_metric_value(box, "map50"), 6),
        "map50_95": round(_metric_value(box, "map"), 6),
        "avg_inference_time_ms": round(float(speed.get("inference", 0.0)), 3),
        "status": "success",
        "message": "Evaluation YOLO terminee.",
        "charts": charts,
        "run_dir": str(save_dir),
    }
    _save_metrics(metrics, output_dir)
    return metrics


def _save_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "yolo_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = output_dir / "metrics_summary.csv"
    fields = [
        "precision",
        "recall",
        "f1_score",
        "map50",
        "map50_95",
        "avg_inference_time_ms",
        "model_path",
        "dataset",
        "status",
        "message",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluer le modele YOLOv8 barcode.")
    parser.add_argument("--model", default="models/barcode_yolov8.pt")
    parser.add_argument("--data", default="data/barcode_dataset/data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_yolo_model(args.model, args.data, args.split, args.imgsz),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
