from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from PIL import Image, ImageOps

try:
    from .image_filter_dl import filter_images, is_barcode_dominant
    from .image_search import search_and_download
    from .improve_images import create_product_video, enhance_image, improve_images_batch
    from .export_marketplace import export_marketplace_package
    from .product_description_generator import save_product_data
    from .product_lookup import lookup_product
    from .scan_qr_code import scan_code
    from .barcode_decoder import decode_barcode
    from .serpapi_product_search import enrich_product_from_barcode
    from .yolo_barcode_detector import detect_barcodes
except ImportError:
    from image_filter_dl import filter_images, is_barcode_dominant
    from image_search import search_and_download
    from improve_images import create_product_video, enhance_image, improve_images_batch
    from export_marketplace import export_marketplace_package
    from product_description_generator import save_product_data
    from product_lookup import lookup_product
    from scan_qr_code import scan_code
    from barcode_decoder import decode_barcode
    from serpapi_product_search import enrich_product_from_barcode
    from yolo_barcode_detector import detect_barcodes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "main"
UPLOADS_DIR = PROJECT_ROOT / "output" / "uploads"
EXPORTS_DIR = PROJECT_ROOT / "output" / "exports"
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_HISTORY_PATH = DATA_DIR / "products_history.json"
MIN_FINAL_IMAGES = 3


def _empty_barcode_metadata(use_yolo: bool) -> dict[str, Any]:
    return {
        "barcode_detection_method": "yolo" if use_yolo else "full_image",
        "barcode_decoding_method": "",
        "barcode_type": "",
        "yolo_boxes": [],
        "yolo_confidence": 0.0,
        "yolo_cropped_images": [],
        "yolo_annotated_image": "",
        "yolo_inference_time_ms": 0.0,
        "barcode_decode_success": False,
        "barcode_decode_error": "",
    }


@dataclass(frozen=True)
class ProductDirs:
    web_images: Path
    filtered_images: Path
    final_images: Path
    processed_original: Path
    videos: Path


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    print(message)
    if callback:
        callback(message)


def sanitize_barcode(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", str(value).strip())
    return cleaned or "unknown"


def get_product_folder(barcode: str) -> str:
    return f"product_{sanitize_barcode(barcode)}"


def build_product_dirs(barcode: str) -> ProductDirs:
    product_folder = get_product_folder(barcode)
    return ProductDirs(
        web_images=MAIN_DIR / "web_images" / product_folder,
        filtered_images=MAIN_DIR / "web_images_filtered" / product_folder,
        final_images=MAIN_DIR / "final_images" / product_folder,
        processed_original=MAIN_DIR / "processed_original" / product_folder,
        videos=MAIN_DIR / "videos" / product_folder,
    )


def ensure_product_dirs(dirs: ProductDirs, clean: bool = False) -> None:
    for directory in asdict(dirs).values():
        path = Path(directory)
        if clean and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def save_original_image(input_image: str | Path, processed_original_dir: Path) -> str:
    output_path = processed_original_dir / "original.jpg"
    processed_original_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_image) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(output_path, "JPEG", quality=95, optimize=True)

    return str(output_path)


def _save_image_file_for_processing(image_file: Any) -> Path:
    if isinstance(image_file, (str, Path)):
        return Path(image_file)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = Path(getattr(image_file, "name", "barcode_upload.jpg")).name
    safe_name = raw_name or "barcode_upload.jpg"
    output_path = UPLOADS_DIR / f"api_{safe_name}"

    if hasattr(image_file, "getbuffer"):
        output_path.write_bytes(bytes(image_file.getbuffer()))
        return output_path

    data = image_file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    output_path.write_bytes(data)
    return output_path


def process_barcode_image(image_file: Any) -> dict[str, Any]:
    """Reusable lightweight barcode service: YOLO detection, pyzbar decode, SerpAPI enrichment."""
    warnings: list[str] = []
    image_path = _save_image_file_for_processing(image_file)
    barcode_metadata = _empty_barcode_metadata(use_yolo=True)

    if not image_path.exists():
        return {
            "status": "error",
            "success": False,
            "message": f"Image introuvable: {image_path}",
            "mode": "YOLOv8 + pyzbar + SerpAPI",
            "barcode": "",
            "warnings": warnings,
            **barcode_metadata,
        }

    try:
        detection = detect_barcodes(image_path)
    except Exception as exc:
        detection = {
            "boxes": [],
            "confidence": 0.0,
            "cropped_image_paths": [],
            "annotated_image_path": "",
            "inference_time_ms": 0.0,
            "warning": "",
            "error": f"YOLO indisponible: {exc}",
        }

    barcode_metadata.update(
        {
            "yolo_boxes": detection.get("boxes", []),
            "yolo_confidence": detection.get("confidence", 0.0),
            "yolo_cropped_images": detection.get("cropped_image_paths", []),
            "yolo_annotated_image": detection.get("annotated_image_path", ""),
            "yolo_inference_time_ms": detection.get("inference_time_ms", 0.0),
        }
    )
    if detection.get("warning"):
        warnings.append(str(detection["warning"]))
    if detection.get("error"):
        warnings.append(str(detection["error"]))

    try:
        decoded = decode_barcode(image_path, detection.get("cropped_image_paths", []))
    except Exception as exc:
        decoded = {
            "barcode_value": "",
            "barcode_type": "",
            "decoding_method": "full_image_zbar",
            "decoding_success": False,
            "error_message": f"Erreur de decodage: {exc}",
        }

    barcode = decoded.get("barcode_value") or ""
    barcode_metadata.update(
        {
            "barcode_detection_method": "yolo" if detection.get("boxes") else "full_image_fallback",
            "barcode_decoding_method": decoded.get("decoding_method", ""),
            "barcode_type": decoded.get("barcode_type", ""),
            "barcode_decode_success": bool(decoded.get("decoding_success")),
            "barcode_decode_error": decoded.get("error_message", ""),
        }
    )

    if not barcode:
        try:
            barcode = scan_code(image_path) or ""
        except Exception as exc:
            warnings.append(f"Fallback scanner historique indisponible: {exc}")
        if barcode:
            barcode_metadata.update(
                {
                    "barcode_detection_method": "legacy_full_image",
                    "barcode_decoding_method": "full_image_zbar",
                    "barcode_decode_success": True,
                    "barcode_decode_error": "",
                }
            )

    if not barcode:
        if decoded.get("error_message"):
            warnings.append(str(decoded["error_message"]))
        return {
            "status": "error",
            "success": False,
            "message": "Aucun code-barres detecte.",
            "mode": "YOLOv8 + pyzbar + SerpAPI",
            "barcode": "",
            "warnings": warnings,
            **barcode_metadata,
        }

    barcode = sanitize_barcode(barcode)
    try:
        serpapi_result = enrich_product_from_barcode(barcode)
    except Exception as exc:
        serpapi_result = {
            "success": False,
            "barcode": barcode,
            "product_title": f"Produit détecté - {barcode}",
            "product_description": "",
            "images": [],
            "links": [],
            "source": "SerpAPI",
            "warning": f"SerpAPI indisponible: {exc}",
        }
    if serpapi_result.get("warning"):
        warnings.append(str(serpapi_result["warning"]))

    return {
        "status": "success",
        "mode": "YOLOv8 + pyzbar + SerpAPI",
        "barcode": barcode,
        "warnings": warnings,
        "serpapi_result": serpapi_result,
        **serpapi_result,
        **barcode_metadata,
    }


def _copy_missing_filtered_sources(
    downloaded_images: list[str],
    selected_images: list[str],
    filtered_dir: Path,
    min_images: int,
) -> list[str]:
    selected = list(selected_images)
    selected_names = {Path(path).name for path in selected}

    for image_path in downloaded_images:
        if len(selected) >= min_images:
            break

        source = Path(image_path)
        if not source.exists() or source.name in selected_names:
            continue
        if is_barcode_dominant(source):
            continue

        target = filtered_dir / source.name
        shutil.copy2(source, target)
        selected.append(str(target))
        selected_names.add(source.name)

    return selected


def build_image_search_query(product: dict[str, str]) -> str:
    parts = [
        product.get("brand", ""),
        product.get("title", ""),
        product.get("quantity", ""),
        product.get("barcode", ""),
    ]
    query = " ".join(part.strip() for part in parts if part and part.strip())
    return query or product.get("barcode", "")


def load_products_history() -> list[dict[str, Any]]:
    if not PRODUCTS_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(PRODUCTS_HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def save_product_history(entry: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_products_history()
    history.insert(0, entry)
    PRODUCTS_HISTORY_PATH.write_text(json.dumps(history[:100], indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    input_image: str | Path,
    max_images: int = 30,
    min_final_images: int = MIN_FINAL_IMAGES,
    clean: bool = True,
    use_yolo: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_image)
    warnings: list[str] = []
    serpapi_result: dict[str, Any] = {}
    barcode_metadata = _empty_barcode_metadata(use_yolo)

    if not input_path.exists():
        return {
            "status": "error",
            "message": f"Image introuvable: {input_path}",
            "warnings": warnings,
            **barcode_metadata,
        }

    barcode: str | None = None
    if use_yolo:
        _notify(progress_callback, "Localisation du code-barres avec YOLOv8...")
        try:
            detection = detect_barcodes(input_path)
        except Exception as exc:
            detection = {
                "boxes": [],
                "confidence": 0.0,
                "cropped_image_paths": [],
                "annotated_image_path": "",
                "inference_time_ms": 0.0,
                "warning": "",
                "error": f"YOLO indisponible: {exc}",
            }

        barcode_metadata.update(
            {
                "yolo_boxes": detection.get("boxes", []),
                "yolo_confidence": detection.get("confidence", 0.0),
                "yolo_cropped_images": detection.get("cropped_image_paths", []),
                "yolo_annotated_image": detection.get("annotated_image_path", ""),
                "yolo_inference_time_ms": detection.get("inference_time_ms", 0.0),
            }
        )
        if detection.get("warning"):
            warnings.append(str(detection["warning"]))
        if detection.get("error"):
            warnings.append(str(detection["error"]))

        _notify(progress_callback, "Decodage ZBar des crops YOLO puis fallback image complete...")
        try:
            decoded = decode_barcode(input_path, detection.get("cropped_image_paths", []))
        except Exception as exc:
            decoded = {
                "barcode_value": "",
                "barcode_type": "",
                "decoding_method": "full_image_zbar",
                "decoding_success": False,
                "error_message": f"Erreur de decodage: {exc}",
            }
        barcode = decoded.get("barcode_value") or None
        barcode_metadata.update(
            {
                "barcode_detection_method": "yolo" if detection.get("boxes") else "full_image_fallback",
                "barcode_decoding_method": decoded.get("decoding_method", ""),
                "barcode_type": decoded.get("barcode_type", ""),
                "barcode_decode_success": bool(decoded.get("decoding_success")),
                "barcode_decode_error": decoded.get("error_message", ""),
            }
        )
        if decoded.get("error_message") and not barcode:
            warnings.append(str(decoded["error_message"]))

    if not barcode:
        _notify(progress_callback, "Fallback vers le scanner historique pyzbar/ZBar...")
        try:
            barcode = scan_code(input_path)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Erreur pendant le scan du code-barres: {exc}",
                "warnings": warnings,
                **barcode_metadata,
            }
        if barcode:
            barcode_metadata.update(
                {
                    "barcode_detection_method": "legacy_full_image",
                    "barcode_decoding_method": "full_image_zbar",
                    "barcode_decode_success": True,
                    "barcode_decode_error": "",
                }
            )

    if not barcode:
        return {
            "status": "error",
            "message": "Aucun code-barres detecte. Verifie zbar/pyzbar et la nettete de l'image.",
            "barcode": None,
            "warnings": warnings,
            **barcode_metadata,
        }

    barcode = sanitize_barcode(barcode)
    product_folder = get_product_folder(barcode)
    product_dirs = build_product_dirs(barcode)
    ensure_product_dirs(product_dirs, clean=clean)

    _notify(progress_callback, "Enrichissement produit via SerpAPI...")
    try:
        serpapi_result = enrich_product_from_barcode(barcode)
    except Exception as exc:
        serpapi_result = {
            "success": False,
            "barcode": barcode,
            "product_title": f"Produit détecté - {barcode}",
            "product_description": "",
            "images": [],
            "links": [],
            "source": "SerpAPI",
            "warning": f"SerpAPI indisponible: {exc}",
        }
    if serpapi_result.get("warning"):
        warnings.append(str(serpapi_result["warning"]))

    _notify(progress_callback, "Identification produit via OpenFoodFacts...")
    try:
        product_lookup = lookup_product(barcode)
    except Exception as exc:
        product_lookup = {
            "barcode": barcode,
            "title": f"Produit {barcode}",
            "brand": "",
            "category": "Categorie provisoire",
            "description": "Description provisoire a valider par le vendeur",
            "quantity": "",
            "image_url": "",
            "source": "fallback",
        }
        warnings.append(f"OpenFoodFacts indisponible: {exc}")
    if product_lookup.get("source") == "fallback":
        warnings.append("Produit non trouve dans OpenFoodFacts. Une fiche provisoire a ete creee.")
    _notify(progress_callback, f"Produit detecte: {product_lookup.get('title', product_folder)}")
    original_processed = save_original_image(input_path, product_dirs.processed_original)

    _notify(progress_callback, "Recherche et telechargement des images web...")
    try:
        downloaded_images = search_and_download(
            barcode=barcode,
            output_dir=product_dirs.web_images,
            max_images=max_images,
            query=build_image_search_query(product_lookup),
        )
    except Exception as exc:
        downloaded_images = []
        warnings.append(f"Recherche web ignoree: {exc}")

    if not downloaded_images:
        warnings.append("Aucune image web telechargee. Verifie SERPAPI_API_KEY ou le quota SerpAPI.")

    _notify(progress_callback, "Filtrage intelligent des images...")
    try:
        selected_images = filter_images(
            original_image=[],
            web_images_dir=product_dirs.web_images,
            filtered_dir=product_dirs.filtered_images,
            threshold=0.65,
            min_images=min_final_images,
            reject_barcode_images=True,
        )
    except Exception as exc:
        selected_images = []
        warnings.append(f"Filtrage IA ignore: {exc}")

    selected_images = _copy_missing_filtered_sources(
        downloaded_images=downloaded_images,
        selected_images=selected_images,
        filtered_dir=product_dirs.filtered_images,
        min_images=min_final_images,
    )
    if len(selected_images) < min_final_images:
        warnings.append(
            "Moins de 3 images produit valides apres filtrage. Verifie la cle SERPAPI_API_KEY "
            "ou relance avec une meilleure image de code-barres."
        )

    _notify(progress_callback, "Suppression d'arriere-plan et amelioration marketplace...")
    try:
        final_images = improve_images_batch(
            image_paths=selected_images,
            final_dir=product_dirs.final_images,
            min_images=min_final_images,
            size=(1000, 1000),
        )
    except Exception as exc:
        final_images = []
        warnings.append(f"Amelioration des images impossible: {exc}")

    if not final_images:
        return {
            "status": "error",
            "message": (
                "Impossible de produire des images marketplace: aucune image produit valide. "
                "Active SerpAPI ou verifie que la recherche web trouve des images du produit."
            ),
            "barcode": barcode,
            "product_folder": product_folder,
            "metadata": product_lookup,
            "product_lookup": product_lookup,
            "serpapi_result": serpapi_result,
            "downloaded_images": downloaded_images,
            "selected_images": selected_images,
            "processed_original_image": original_processed,
            "warnings": warnings,
            **barcode_metadata,
        }

    try:
        video_path = create_product_video(
            final_images,
            product_dirs.videos / "product_showcase.mp4",
            title=product_lookup.get("title"),
        )
    except Exception as exc:
        video_path = None
        warnings.append(f"Generation video impossible: {exc}")
    if not video_path:
        warnings.append("Video courte non generee. Les images finales restent disponibles.")

    _notify(progress_callback, "Generation de la fiche produit...")
    export_dir = EXPORTS_DIR / product_folder
    product_data_path = export_dir / "product_data.json"
    product_data = save_product_data(product_lookup, product_data_path)
    product_data.update(
        {
            "barcode_detection_method": barcode_metadata["barcode_detection_method"],
            "barcode_decoding_method": barcode_metadata["barcode_decoding_method"],
            "yolo_boxes": barcode_metadata["yolo_boxes"],
            "yolo_confidence": barcode_metadata["yolo_confidence"],
            "yolo_annotated_image": barcode_metadata["yolo_annotated_image"],
            "yolo_inference_time_ms": barcode_metadata["yolo_inference_time_ms"],
            "barcode_decode_success": barcode_metadata["barcode_decode_success"],
        }
    )
    product_data_path.write_text(
        json.dumps(product_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _notify(progress_callback, "Export JSON, CSV et ZIP marketplace...")
    try:
        export_paths = export_marketplace_package(
            product_data=product_data,
            final_images=final_images,
            video_path=video_path,
            export_dir=export_dir,
        )
    except Exception as exc:
        export_paths = {}
        warnings.append(f"Export marketplace non genere: {exc}")

    history_entry = {
        "date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "barcode": barcode,
        "title": product_data.get("title", product_lookup.get("title", "")),
        "brand": product_data.get("brand", ""),
        "category": product_data.get("marketplace_category") or product_data.get("category", ""),
        "final_images_count": len(final_images),
        "video_path": video_path or "",
        "export_zip": export_paths.get("zip_path", ""),
    }
    save_product_history(history_entry)

    result = {
        "status": "success",
        "message": "Pipeline WAHA IA termine",
        "barcode": barcode,
        "product_folder": product_folder,
        "metadata": product_lookup,
        "product_lookup": product_lookup,
        "serpapi_result": serpapi_result,
        "product_data": product_data,
        "product_data_path": str(product_data_path),
        "downloaded_images_count": len(downloaded_images),
        "selected_images_count": len(selected_images),
        "final_images_count": len(final_images),
        "downloaded_images": downloaded_images,
        "selected_images": selected_images,
        "final_images": final_images,
        "video_path": video_path,
        "processed_original_image": original_processed,
        "web_images_dir": str(product_dirs.web_images),
        "filtered_images_dir": str(product_dirs.filtered_images),
        "final_images_dir": str(product_dirs.final_images),
        "processed_original_dir": str(product_dirs.processed_original),
        "videos_dir": str(product_dirs.videos),
        "export_paths": export_paths,
        "export_dir": export_paths.get("export_dir", str(export_dir)),
        "export_zip": export_paths.get("zip_path", ""),
        "history_path": str(PRODUCTS_HISTORY_PATH),
        "warnings": warnings,
        **barcode_metadata,
    }
    _notify(progress_callback, "Pipeline termine.")
    return result


def clean_all_results() -> None:
    for base_dir in [
        MAIN_DIR / "web_images",
        MAIN_DIR / "web_images_filtered",
        MAIN_DIR / "final_images",
        MAIN_DIR / "processed_original",
        MAIN_DIR / "videos",
        MAIN_DIR / "yolo_detections",
        EXPORTS_DIR,
    ]:
        base_dir.mkdir(parents=True, exist_ok=True)
        for item in base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline IA WAHA: barcode vers images e-commerce.")
    parser.add_argument("image", help="Image contenant le code-barres")
    parser.add_argument("--max-images", type=int, default=30, help="Nombre maximal d'images web")
    parser.add_argument("--min-final", type=int, default=3, help="Nombre minimal d'images finales")
    parser.add_argument("--no-clean", action="store_true", help="Conserver les anciens fichiers du produit")
    parser.add_argument("--no-yolo", action="store_true", help="Desactiver YOLO et utiliser le scanner historique")
    args = parser.parse_args()

    result = run_pipeline(
        input_image=args.image,
        max_images=args.max_images,
        min_final_images=args.min_final,
        clean=not args.no_clean,
        use_yolo=not args.no_yolo,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
