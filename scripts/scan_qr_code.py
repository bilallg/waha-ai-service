from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

try:
    from pyzbar.pyzbar import decode
except Exception:
    decode = None


def _decode_with_pyzbar(image) -> list[str]:
    if decode is None:
        return []

    values: list[str] = []
    for item in decode(image):
        try:
            value = item.data.decode("utf-8").strip()
        except UnicodeDecodeError:
            value = item.data.decode("latin-1", errors="ignore").strip()

        if value and value not in values:
            values.append(value)

    return values


def _preprocess_variants(image) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = [image, gray]

    height, width = gray.shape[:2]
    if max(height, width) < 1200:
        scale = 1200 / max(height, width)
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(resized)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    variants.append(sharpened)

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    variants.append(threshold)

    return variants


def scan_codes(image_path: str | Path) -> list[str]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable: {path}")

    image = cv2.imread(str(path))
    if image is None:
        print(f"Impossible de charger l'image: {path}")
        return []

    if decode is None:
        print(
            "pyzbar/zbar indisponible pour les codes-barres lineaires. "
            "macOS: brew install zbar"
        )
    else:
        for variant in _preprocess_variants(image):
            values = _decode_with_pyzbar(variant)
            if values:
                return values

    qr_detector = cv2.QRCodeDetector()
    value, _, _ = qr_detector.detectAndDecode(image)
    return [value.strip()] if value else []


def scan_code(image_path: str | Path) -> Optional[str]:
    codes = scan_codes(image_path)
    return codes[0] if codes else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scanner un code-barres ou QR code depuis une image.")
    parser.add_argument("image", help="Chemin de l'image a analyser")
    args = parser.parse_args()

    code = scan_code(args.image)
    print(code or "Aucun code detecte")
