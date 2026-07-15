from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


NOT_DETECTED_MESSAGE = "Expiration date not detected"
MIN_OCR_CONFIDENCE = 0.35

KEYWORD_PATTERN = (
    r"(?:exp(?:iry|iration)?|best\s*before|use\s*by|bb|b\.b\.|"
    r"date\s*limite|dlc|a\s*consommer\s*avant|à\s*consommer\s*avant)"
)

DATE_PATTERNS = [
    re.compile(
        rf"(?P<context>{KEYWORD_PATTERN})?\s*:?\s*(?P<date>\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<context>{KEYWORD_PATTERN})?\s*:?\s*(?P<date>\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<context>{KEYWORD_PATTERN})?\s*:?\s*(?P<date>\d{{1,2}}[-/.]\d{{2,4}})",
        re.IGNORECASE,
    ),
]


def _not_detected() -> dict[str, Any]:
    return {
        "expiration_date": None,
        "expiration_text": "",
        "expiration_confidence": None,
        "expiration_found": False,
        "message": NOT_DETECTED_MESSAGE,
    }


def _normalize_text(text: str) -> str:
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _parse_confidence(value: object) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0:
        return None
    return max(0.0, min(confidence / 100.0, 1.0))


def _expand_year(year: str) -> int:
    numeric = int(year)
    if len(year) == 2:
        return 2000 + numeric
    return numeric


def _valid_day(year: int, month: int, day: int) -> bool:
    try:
        datetime(year, month, day)
    except ValueError:
        return False
    return True


def _normalize_date(raw_date: str) -> str | None:
    cleaned = raw_date.strip().replace(".", "/")
    separator = "-" if "-" in cleaned else "/"
    parts = [part for part in re.split(r"[-/]", cleaned) if part]

    if len(parts) == 3 and len(parts[0]) == 4:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        if _valid_day(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    if len(parts) == 3:
        day = int(parts[0])
        month = int(parts[1])
        year = _expand_year(parts[2])
        if _valid_day(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    if len(parts) == 2:
        month = int(parts[0])
        year = _expand_year(parts[1])
        if 1 <= month <= 12 and 2000 <= year <= 2099:
            return f"{year:04d}-{month:02d}"

    return None


def _match_expiration_date(text: str) -> tuple[str | None, str]:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return None, ""

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(normalized_text):
            raw_date = match.group("date")
            normalized_date = _normalize_date(raw_date)
            if normalized_date:
                return normalized_date, match.group(0).strip(" :;-")

    return None, ""


def _preprocess_images(image_path: str | Path) -> list[Image.Image]:
    with Image.open(image_path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")

    grayscale = ImageOps.grayscale(original)
    scale = 2 if max(grayscale.size) < 1800 else 1
    resized = grayscale.resize((grayscale.width * scale, grayscale.height * scale))
    contrasted = ImageEnhance.Contrast(ImageOps.autocontrast(resized)).enhance(1.8)
    threshold = contrasted.point(lambda pixel: 255 if pixel > 155 else 0)

    return [contrasted, threshold, resized, grayscale, original]


def _ocr_image(image: Image.Image) -> tuple[str, float | None]:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed") from exc

    last_error: Exception | None = None
    data: dict[str, Any] | None = None
    for language in ("eng+fra", "eng", None):
        try:
            kwargs = {
                "config": "--psm 6",
                "output_type": pytesseract.Output.DICT,
            }
            if language:
                kwargs["lang"] = language
            data = pytesseract.image_to_data(image, **kwargs)
            break
        except Exception as exc:
            last_error = exc
    if data is None:
        raise RuntimeError(f"Tesseract OCR failed: {last_error}")

    words: list[str] = []
    confidences: list[float] = []

    for word, confidence_value in zip(data.get("text", []), data.get("conf", [])):
        cleaned_word = str(word or "").strip()
        if not cleaned_word:
            continue
        words.append(cleaned_word)
        confidence = _parse_confidence(confidence_value)
        if confidence is not None:
            confidences.append(confidence)

    text = _normalize_text(" ".join(words))
    average_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
    return text, average_confidence


def detect_expiration_date(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return _not_detected()

    try:
        variants = _preprocess_images(path)
    except Exception:
        return _not_detected()

    try:
        for image in variants:
            text, confidence = _ocr_image(image)

            expiration_date, expiration_text = _match_expiration_date(text)
            if not expiration_date:
                continue
            if confidence is not None and confidence < MIN_OCR_CONFIDENCE:
                continue

            return {
                "expiration_date": expiration_date,
                "expiration_text": expiration_text,
                "expiration_confidence": confidence,
                "expiration_found": True,
                "message": "Expiration date detected",
            }
    except Exception:
        return _not_detected()

    return _not_detected()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect expiration date from a product image with OCR.")
    parser.add_argument("image", help="Image containing product packaging text")
    args = parser.parse_args()

    print(json.dumps(detect_expiration_date(args.image), indent=2, ensure_ascii=False))
