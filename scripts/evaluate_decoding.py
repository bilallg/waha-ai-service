from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

try:
    from .barcode_decoder import decode_barcode, decode_full_image
    from .yolo_barcode_detector import DEFAULT_MODEL_PATH, detect_barcodes
except ImportError:
    from barcode_decoder import decode_barcode, decode_full_image
    from yolo_barcode_detector import DEFAULT_MODEL_PATH, detect_barcodes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "output" / "evaluation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_test_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def evaluate_decoding_folder(
    test_folder: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    images = collect_test_images(test_folder)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    detected_by_yolo = 0
    successfully_decoded = 0
    failed_images: list[str] = []
    decode_times: list[float] = []

    for image_path in images:
        detection = detect_barcodes(image_path, model_path=model_path)
        if detection.get("boxes"):
            detected_by_yolo += 1
        decoded = decode_barcode(image_path, detection.get("cropped_image_paths", []))
        success = bool(decoded.get("decoding_success"))
        if success:
            successfully_decoded += 1
        else:
            failed_images.append(str(image_path))
        decode_times.append(float(decoded.get("decode_time_ms", 0.0)))
        rows.append(
            {
                "image": str(image_path),
                "yolo_detected": bool(detection.get("boxes")),
                "yolo_confidence": detection.get("confidence", 0.0),
                "decoded": success,
                "barcode_value": decoded.get("barcode_value", ""),
                "barcode_type": decoded.get("barcode_type", ""),
                "method": decoded.get("decoding_method", ""),
                "yolo_inference_time_ms": detection.get("inference_time_ms", 0.0),
                "decode_time_ms": decoded.get("decode_time_ms", 0.0),
                "error": decoded.get("error_message") or detection.get("error", ""),
            }
        )

    total = len(images)
    metrics = {
        "total_images": total,
        "detected_by_yolo": detected_by_yolo,
        "successfully_decoded": successfully_decoded,
        "detection_rate": round(detected_by_yolo / total, 6) if total else 0.0,
        "decoding_success_rate": round(successfully_decoded / total, 6) if total else 0.0,
        "avg_decode_time_ms": round(sum(decode_times) / len(decode_times), 3) if decode_times else 0.0,
        "failed_images": failed_images,
        "method": "YOLOv8 + ZBar with full-image fallback",
        "test_folder": str(test_folder),
        "status": "success" if total else "error",
        "message": "" if total else f"Aucune image de test trouvee dans {test_folder}",
    }
    (EVALUATION_DIR / "decoding_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_rows(EVALUATION_DIR / "decoding_results.csv", rows)
    return metrics


def compare_decoding_methods(
    test_folder: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    images = collect_test_images(test_folder)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    old_success = 0
    new_success = 0
    old_times: list[float] = []
    new_times: list[float] = []
    old_failed: list[str] = []
    new_failed: list[str] = []

    for image_path in images:
        old_started = time.perf_counter()
        old_result = decode_full_image(image_path)
        old_elapsed = (time.perf_counter() - old_started) * 1000

        new_started = time.perf_counter()
        detection = detect_barcodes(image_path, model_path=model_path)
        new_result = decode_barcode(image_path, detection.get("cropped_image_paths", []))
        new_elapsed = (time.perf_counter() - new_started) * 1000

        old_ok = bool(old_result.get("decoding_success"))
        new_ok = bool(new_result.get("decoding_success"))
        old_success += int(old_ok)
        new_success += int(new_ok)
        old_times.append(old_elapsed)
        new_times.append(new_elapsed)
        if not old_ok:
            old_failed.append(str(image_path))
        if not new_ok:
            new_failed.append(str(image_path))
        rows.append(
            {
                "image": str(image_path),
                "old_success": old_ok,
                "new_success": new_ok,
                "old_time_ms": round(old_elapsed, 3),
                "new_time_ms": round(new_elapsed, 3),
                "new_method": new_result.get("decoding_method", ""),
            }
        )

    total = len(images)
    comparison = {
        "total_images": total,
        "old_method": {
            "name": "pyzbar/ZBar full image",
            "successful_scans": old_success,
            "success_rate": round(old_success / total, 6) if total else 0.0,
            "average_time_ms": round(sum(old_times) / len(old_times), 3) if old_times else 0.0,
            "failed_images": old_failed,
        },
        "new_method": {
            "name": "YOLOv8 + ZBar + full-image fallback",
            "successful_scans": new_success,
            "success_rate": round(new_success / total, 6) if total else 0.0,
            "average_time_ms": round(sum(new_times) / len(new_times), 3) if new_times else 0.0,
            "failed_images": new_failed,
        },
        "status": "success" if total else "error",
        "message": "" if total else f"Aucune image de test trouvee dans {test_folder}",
    }
    (EVALUATION_DIR / "decoding_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_rows(EVALUATION_DIR / "decoding_comparison.csv", rows)
    return comparison


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluer le taux de decodage des codes-barres.")
    parser.add_argument("test_folder", help="Dossier d'images de test")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--compare", action="store_true", help="Comparer ancien et nouveau scan")
    args = parser.parse_args()
    payload = (
        compare_decoding_methods(args.test_folder, args.model)
        if args.compare
        else evaluate_decoding_folder(args.test_folder, args.model)
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
