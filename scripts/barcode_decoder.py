from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
MAX_PREPROCESS_SIDE = 4200

try:
    from pyzbar.pyzbar import decode as zbar_decode

    ZBAR_IMPORT_ERROR = ""
except Exception as exc:
    zbar_decode = None
    ZBAR_IMPORT_ERROR = str(exc)


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variants: list[tuple[str, np.ndarray]] = [("original", image), ("grayscale", gray)]

    for scale in (2.0, 3.0):
        max_side = max(gray.shape[:2])
        effective_scale = min(scale, MAX_PREPROCESS_SIDE / max_side) if max_side else scale
        if effective_scale <= 1.05:
            resized = gray
            scale_label = "1x"
        else:
            resized = cv2.resize(gray, None, fx=effective_scale, fy=effective_scale, interpolation=cv2.INTER_CUBIC)
            scale_label = f"{effective_scale:.2f}x"
        variants.append((f"grayscale_{scale_label}", resized))

        equalized = cv2.equalizeHist(resized)
        variants.append((f"equalized_{scale_label}", equalized))

        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        sharpened = cv2.addWeighted(resized, 1.9, blurred, -0.9, 0)
        variants.append((f"sharpened_{scale_label}", sharpened))

        high_contrast = cv2.convertScaleAbs(sharpened, alpha=1.45, beta=0)
        variants.append((f"contrast_{scale_label}", high_contrast))

        for threshold_value in (95, 125, 155):
            _, thresholded = cv2.threshold(high_contrast, threshold_value, 255, cv2.THRESH_BINARY)
            variants.append((f"threshold_{threshold_value}_{scale_label}", thresholded))

        variants.append(
            (
                f"adaptive_threshold_{scale_label}",
                cv2.adaptiveThreshold(
                    high_contrast,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    5,
                ),
            )
        )

    return variants


def _decode_image(image: np.ndarray) -> tuple[str, str, str, int, str]:
    decoded_count = 0
    for rotation in (0, 90, 180, 270):
        rotated = _rotate(image, rotation)
        for variant_name, variant in _variants(rotated):
            decoded_items = zbar_decode(variant)
            decoded_count += len(decoded_items or [])
            if not decoded_items:
                continue
            item = decoded_items[0]
            try:
                value = item.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                value = item.data.decode("latin-1", errors="ignore").strip()
            if value:
                return value, str(item.type or ""), "", decoded_count, f"{variant_name}_rot{rotation}"
    return "", "", "Aucun code-barres decode.", decoded_count, ""


def _decode_path(image_path: str | Path) -> tuple[str, str, str, int, str]:
    if zbar_decode is None:
        return "", "", f"pyzbar/ZBar indisponible: {ZBAR_IMPORT_ERROR or 'installation manquante'}", 0, ""

    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return "", "", f"Image illisible: {path}", 0, ""

    try:
        return _decode_image(image)
    except Exception as exc:
        return "", "", f"Erreur ZBar: {exc}", 0, ""


def decode_barcode(
    full_image_path: str | Path,
    cropped_image_paths: list[str | Path] | None = None,
    full_image_first: bool = True,
) -> dict[str, Any]:
    """Decode the complete image and YOLO crops using preprocessing plus rotations."""
    started_at = time.perf_counter()
    errors: list[str] = []
    decoded_count = 0

    decode_targets: list[tuple[str, str | Path]] = []
    if full_image_first:
        decode_targets.append(("full_image_zbar", full_image_path))
    decode_targets.extend(("yolo_crop_zbar", crop_path) for crop_path in (cropped_image_paths or []))
    if not full_image_first:
        decode_targets.append(("full_image_zbar", full_image_path))

    for method, image_path in decode_targets:
        value, barcode_type, error, attempt_decoded_count, variant_method = _decode_path(image_path)
        decoded_count += attempt_decoded_count
        if value:
            return {
                "barcode_value": value,
                "barcode_type": barcode_type,
                "decoding_method": method,
                "decoding_variant": variant_method,
                "decoding_success": True,
                "error_message": "",
                "decoded_image_path": str(image_path),
                "pyzbar_decoded_count": decoded_count,
                "decode_time_ms": round((time.perf_counter() - started_at) * 1000, 3),
            }
        if error:
            errors.append(f"{Path(image_path).name}: {error}")

    message = "; ".join(dict.fromkeys(errors)) or "Aucun code-barres decode."
    LOGGER.warning(message)
    return {
        "barcode_value": "",
        "barcode_type": "",
        "decoding_method": "full_image_zbar",
        "decoding_variant": "",
        "decoding_success": False,
        "error_message": message,
        "decoded_image_path": "",
        "pyzbar_decoded_count": decoded_count,
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
