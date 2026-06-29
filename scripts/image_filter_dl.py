from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _valid_image_paths(folder: str | Path) -> list[Path]:
    path = Path(folder)
    if not path.exists():
        return []

    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_barcode_dominant(image_path: str | Path) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False

    if zbar_decode is not None:
        variants = [image]
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        variants.append(cv2.addWeighted(image, 1.6, blurred, -0.6, 0))
        variants.append(
            cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            )
        )
        decoded_items = []
        for variant in variants:
            decoded_items = zbar_decode(variant)
            if decoded_items:
                break

        if decoded_items:
            image_height, image_width = image.shape[:2]
            image_area = max(image_height * image_width, 1)
            for item in decoded_items:
                rect = item.rect
                rect_area_ratio = (rect.width * rect.height) / image_area
                center_x = rect.left + rect.width / 2
                center_y = rect.top + rect.height / 2
                central = (
                    image_width * 0.22 <= center_x <= image_width * 0.78
                    and image_height * 0.22 <= center_y <= image_height * 0.78
                )
                if rect_area_ratio > 0.035 or (central and rect_area_ratio > 0.015):
                    return True

    qr_detector = cv2.QRCodeDetector()
    detected, points = qr_detector.detect(image)
    if detected and points is not None:
        image_area = image.shape[0] * image.shape[1]
        qr_area = float(cv2.contourArea(points.reshape(-1, 2).astype("float32")))
        if image_area > 0 and qr_area / image_area > 0.18:
            return True

    height, width = image.shape[:2]
    scale = 700 / max(height, width) if max(height, width) > 700 else 1.0
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = float((binary < 80).mean())
    white_ratio = float((binary > 220).mean())
    vertical_transitions = float((abs(binary[:, 1:].astype("int16") - binary[:, :-1].astype("int16")) > 0).mean())
    vertical_edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    vertical_edge_density = float((abs(vertical_edges) > 70).mean())

    return (
        white_ratio > 0.45
        and 0.03 < black_ratio < 0.55
        and vertical_transitions > 0.10
        and vertical_edge_density > 0.06
    )


@lru_cache(maxsize=1)
def _load_deep_feature_extractor():
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
    except Exception as exc:
        print(f"Filtrage DL indisponible, fallback qualite: {exc}")
        return None

    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_model = nn.Sequential(*list(model.children())[:-1])
        feature_model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        return {
            "torch": torch,
            "transform": transform,
            "model": feature_model,
            "cosine": torch.nn.CosineSimilarity(dim=1),
        }
    except Exception as exc:
        print(f"Impossible de charger ResNet50, fallback qualite: {exc}")
        return None


def _deep_embedding(image_path: str | Path):
    extractor = _load_deep_feature_extractor()
    if extractor is None:
        return None

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = extractor["transform"](image).unsqueeze(0)

    with extractor["torch"].no_grad():
        embedding = extractor["model"](tensor).flatten(1)

    return embedding


def _deep_similarity(original_image: str | Path, candidate_image: str | Path) -> float | None:
    extractor = _load_deep_feature_extractor()
    if extractor is None:
        return None

    original_embedding = _deep_embedding(original_image)
    candidate_embedding = _deep_embedding(candidate_image)
    if original_embedding is None or candidate_embedding is None:
        return None

    return float(extractor["cosine"](original_embedding, candidate_embedding).item())


def _best_reference_similarity(reference_images: list[Path], candidate_image: Path) -> float | None:
    scores: list[float] = []

    for reference_image in reference_images:
        score = _deep_similarity(reference_image, candidate_image)
        if score is not None:
            scores.append(score)

    return max(scores) if scores else None


def _quality_score(image_path: str | Path) -> float:
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            resolution_score = min((width * height) / (1000 * 1000), 1.0)

            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            histogram = edges.histogram()
            sharpness_score = sum(value * index for index, value in enumerate(histogram)) / (
                255 * max(width * height, 1)
            )

            return min(1.0, 0.65 * resolution_score + 0.35 * sharpness_score)
    except Exception:
        return 0.0


def _copy_selected_images(selected: list[tuple[float, Path]], filtered_dir: Path) -> list[str]:
    filtered_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for index, (score, image_path) in enumerate(selected, start=1):
        output_path = filtered_dir / image_path.name
        if output_path.exists():
            output_path = filtered_dir / f"selected_{index}{image_path.suffix.lower()}"

        try:
            shutil.copy2(image_path, output_path)
            copied.append(str(output_path))
            print(f"Image gardee: {image_path.name} score={score:.3f}")
        except Exception as exc:
            print(f"Copie ignoree pour {image_path}: {exc}")

    return copied


def filter_images(
    original_image: str | Path | list[str | Path],
    web_images_dir: str | Path,
    filtered_dir: str | Path,
    threshold: float = 0.65,
    min_images: int = 3,
    reject_barcode_images: bool = True,
) -> list[str]:
    candidates = _valid_image_paths(web_images_dir)
    filtered_path = Path(filtered_dir)
    reference_images = (
        [Path(path) for path in original_image]
        if isinstance(original_image, list)
        else [Path(original_image)]
    )
    reference_images = [
        path
        for path in reference_images
        if path.exists() and not (reject_barcode_images and is_barcode_dominant(path))
    ]

    if not candidates:
        print("Aucune image web disponible pour le filtrage.")
        return []

    scored: list[tuple[float, Path]] = []
    deep_learning_available = _load_deep_feature_extractor() is not None

    for candidate in candidates:
        try:
            with Image.open(candidate) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Image invalide ignoree {candidate.name}: {exc}")
            continue

        if reject_barcode_images and is_barcode_dominant(candidate):
            print(f"Image barcode dominante ignoree {candidate.name}")
            continue

        if deep_learning_available and reference_images:
            score = _best_reference_similarity(reference_images, candidate)
            if score is None:
                score = _quality_score(candidate)
        else:
            score = _quality_score(candidate)

        scored.append((score, candidate))
        print(f"{candidate.name} score={score:.3f}")

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item for item in scored if item[0] >= threshold]

    if len(selected) < min_images:
        selected = scored[: min(min_images, len(scored))]

    return _copy_selected_images(selected, filtered_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filtrer les images web par similarite visuelle.")
    parser.add_argument("original_image", help="Image originale contenant le produit/code-barres")
    parser.add_argument("web_images_dir", help="Dossier des images telechargees")
    parser.add_argument("filtered_dir", help="Dossier des images filtrees")
    args = parser.parse_args()

    filter_images(args.original_image, args.web_images_dir, args.filtered_dir)
