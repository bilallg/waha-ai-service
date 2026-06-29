from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "barcode_yolov8.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "main" / "yolo_detections"


def _safe_identifier(value: str | None) -> str:
    if value:
        cleaned = "".join(character for character in str(value) if character.isalnum() or character in "_-")
        if cleaned:
            return cleaned
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _empty_result(
    image_path: Path,
    model_path: Path,
    output_dir: Path,
    warning: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "success": False,
        "model_available": model_path.exists(),
        "image_path": str(image_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "boxes": [],
        "confidence": 0.0,
        "class_name": "",
        "cropped_image_paths": [],
        "annotated_image_path": "",
        "inference_time_ms": 0.0,
        "warning": warning,
        "error": error,
    }


def detect_barcodes(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    product_identifier: str | None = None,
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    """Locate barcodes with YOLOv8 and persist crops plus an annotated image."""
    source = Path(image_path)
    weights = Path(model_path)
    output_dir = Path(output_root) / f"product_{_safe_identifier(product_identifier)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        return _empty_result(source, weights, output_dir, error=f"Image introuvable: {source}")

    if not weights.exists():
        warning = (
            f"Modele YOLO absent: {weights}. "
            "Le pipeline utilisera le scan ZBar/pyzbar sur l'image complete."
        )
        LOGGER.warning(warning)
        return _empty_result(source, weights, output_dir, warning=warning)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        warning = f"Ultralytics indisponible ({exc}). Utilisation du fallback ZBar/pyzbar."
        LOGGER.warning(warning)
        return _empty_result(source, weights, output_dir, warning=warning)

    image = cv2.imread(str(source))
    if image is None:
        return _empty_result(source, weights, output_dir, error=f"Image illisible: {source}")

    try:
        model = YOLO(str(weights))
        started_at = time.perf_counter()
        predictions = model.predict(source=str(source), conf=confidence_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
    except Exception as exc:
        error = f"Echec inference YOLO: {exc}"
        LOGGER.exception(error)
        return _empty_result(source, weights, output_dir, error=error)

    if not predictions:
        result = _empty_result(source, weights, output_dir, warning="YOLO n'a retourne aucune prediction.")
        result["inference_time_ms"] = round(elapsed_ms, 3)
        return result

    prediction = predictions[0]
    boxes_payload: list[dict[str, Any]] = []
    crop_paths: list[str] = []
    names = getattr(prediction, "names", {}) or {}
    raw_boxes = getattr(prediction, "boxes", None)

    if raw_boxes is not None:
        for index, box in enumerate(raw_boxes):
            coordinates = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = str(names.get(class_id, class_id))
            left, top, right, bottom = coordinates
            left = max(0, int(round(left)))
            top = max(0, int(round(top)))
            right = min(image.shape[1], int(round(right)))
            bottom = min(image.shape[0], int(round(bottom)))

            if right <= left or bottom <= top:
                continue

            crop = image[top:bottom, left:right]
            crop_path = output_dir / f"barcode_crop_{index + 1}.jpg"
            if crop.size and cv2.imwrite(str(crop_path), crop):
                crop_paths.append(str(crop_path))

            boxes_payload.append(
                {
                    "xyxy": [left, top, right, bottom],
                    "confidence": round(confidence, 6),
                    "class_id": class_id,
                    "class_name": class_name,
                    "crop_path": str(crop_path) if crop_path.exists() else "",
                }
            )

    annotated_path = output_dir / "annotated.jpg"
    try:
        annotated = prediction.plot()
        if annotated is not None:
            cv2.imwrite(str(annotated_path), annotated)
    except Exception as exc:
        LOGGER.warning("Impossible de sauvegarder l'image YOLO annotee: %s", exc)

    best_box = max(boxes_payload, key=lambda item: item["confidence"], default={})
    return {
        "success": bool(boxes_payload),
        "model_available": True,
        "image_path": str(source),
        "model_path": str(weights),
        "output_dir": str(output_dir),
        "boxes": boxes_payload,
        "confidence": float(best_box.get("confidence", 0.0)),
        "class_name": str(best_box.get("class_name", "")),
        "cropped_image_paths": crop_paths,
        "annotated_image_path": str(annotated_path) if annotated_path.exists() else "",
        "inference_time_ms": round(elapsed_ms, 3),
        "warning": "" if boxes_payload else "Aucun code-barres localise par YOLO.",
        "error": "",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detecter un code-barres avec YOLOv8.")
    parser.add_argument("image", help="Image a analyser")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Poids YOLOv8")
    parser.add_argument("--confidence", type=float, default=0.25, help="Seuil de confiance")
    args = parser.parse_args()
    print(
        json.dumps(
            detect_barcodes(args.image, args.model, confidence_threshold=args.confidence),
            indent=2,
            ensure_ascii=False,
        )
    )
