from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2


LOGGER = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import decode as zbar_decode

    ZBAR_IMPORT_ERROR = ""
except Exception as exc:
    zbar_decode = None
    ZBAR_IMPORT_ERROR = str(exc)


def _variants(image) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variants = [image, gray]
    height, width = gray.shape[:2]
    if max(height, width) < 1400:
        scale = 1400 / max(height, width)
        variants.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(cv2.addWeighted(gray, 1.8, blurred, -0.8, 0))
    variants.append(
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
    )
    return variants


def _decode_path(image_path: str | Path) -> tuple[str, str, str]:
    if zbar_decode is None:
        return "", "", f"pyzbar/ZBar indisponible: {ZBAR_IMPORT_ERROR or 'installation manquante'}"

    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return "", "", f"Image illisible: {path}"

    try:
        for variant in _variants(image):
            decoded_items = zbar_decode(variant)
            if not decoded_items:
                continue
            item = decoded_items[0]
            try:
                value = item.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                value = item.data.decode("latin-1", errors="ignore").strip()
            if value:
                return value, str(item.type or ""), ""
    except Exception as exc:
        return "", "", f"Erreur ZBar: {exc}"
    return "", "", "Aucun code-barres decode."


def decode_barcode(
    full_image_path: str | Path,
    cropped_image_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Decode YOLO crops first, then fall back to the complete image."""
    started_at = time.perf_counter()
    errors: list[str] = []

    for crop_path in cropped_image_paths or []:
        value, barcode_type, error = _decode_path(crop_path)
        if value:
            return {
                "barcode_value": value,
                "barcode_type": barcode_type,
                "decoding_method": "yolo_crop_zbar",
                "decoding_success": True,
                "error_message": "",
                "decoded_image_path": str(crop_path),
                "decode_time_ms": round((time.perf_counter() - started_at) * 1000, 3),
            }
        if error:
            errors.append(f"{Path(crop_path).name}: {error}")

    value, barcode_type, error = _decode_path(full_image_path)
    if value:
        return {
            "barcode_value": value,
            "barcode_type": barcode_type,
            "decoding_method": "full_image_zbar",
            "decoding_success": True,
            "error_message": "",
            "decoded_image_path": str(full_image_path),
            "decode_time_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }

    if error:
        errors.append(error)
    message = "; ".join(dict.fromkeys(errors)) or "Aucun code-barres decode."
    LOGGER.warning(message)
    return {
        "barcode_value": "",
        "barcode_type": "",
        "decoding_method": "full_image_zbar",
        "decoding_success": False,
        "error_message": message,
        "decoded_image_path": "",
        "decode_time_ms": round((time.perf_counter() - started_at) * 1000, 3),
    }


def decode_full_image(image_path: str | Path) -> dict[str, Any]:
    return decode_barcode(image_path, [])


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Decoder un code-barres avec ZBar/pyzbar.")
    parser.add_argument("image", help="Image complete")
    parser.add_argument("--crops", nargs="*", default=[], help="Crops YOLO a essayer en premier")
    args = parser.parse_args()
    print(json.dumps(decode_barcode(args.image, args.crops), indent=2, ensure_ascii=False))
