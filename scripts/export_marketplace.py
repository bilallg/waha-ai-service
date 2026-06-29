from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "barcode",
    "title",
    "brand",
    "category",
    "short_description",
    "long_description",
    "tags",
    "image_folder",
    "video_path",
]


def _existing_paths(paths: list[str | Path] | None) -> list[Path]:
    return [Path(path) for path in (paths or []) if Path(path).exists()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, product_data: dict[str, Any], image_folder: str, video_path: str) -> None:
    row = {
        "barcode": product_data.get("barcode", ""),
        "title": product_data.get("seo_title") or product_data.get("title", ""),
        "brand": product_data.get("brand", ""),
        "category": product_data.get("marketplace_category") or product_data.get("category", ""),
        "short_description": product_data.get("short_description", ""),
        "long_description": product_data.get("long_description", ""),
        "tags": ", ".join(product_data.get("tags") or []),
        "image_folder": image_folder,
        "video_path": video_path,
    }
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def _copy_export_assets(export_dir: Path, final_images: list[Path], video_path: Path | None) -> tuple[Path, Path | None]:
    images_dir = export_dir / "images_finales"
    images_dir.mkdir(parents=True, exist_ok=True)
    for image in final_images:
        shutil.copy2(image, images_dir / image.name)

    copied_video: Path | None = None
    if video_path and video_path.exists():
        copied_video = export_dir / "video" / video_path.name
        copied_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, copied_video)

    return images_dir, copied_video


def _zip_export(export_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(export_dir.rglob("*")):
            if item.is_file() and item != zip_path:
                archive.write(item, item.relative_to(export_dir))
    return zip_path


def export_marketplace_package(
    product_data: dict[str, Any],
    final_images: list[str | Path],
    video_path: str | Path | None,
    export_dir: str | Path,
    approval: dict[str, Any] | None = None,
) -> dict[str, str]:
    export_path = Path(export_dir)
    if export_path.exists():
        shutil.rmtree(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    image_paths = _existing_paths(final_images)
    if not image_paths:
        raise ValueError("Aucune image finale valide a exporter.")

    source_video = Path(video_path) if video_path else None
    images_dir, copied_video = _copy_export_assets(export_path, image_paths, source_video)

    product_json = export_path / "product.json"
    product_data_json = export_path / "product_data.json"
    product_csv = export_path / "product.csv"
    approval_json = export_path / "seller_approval.json"
    zip_path = export_path / f"waha_export_{product_data.get('barcode', 'product')}.zip"

    approval_payload = approval or {
        "barcode": product_data.get("barcode", ""),
        "accepted_images": [str(path) for path in image_paths],
        "rejected_images": [],
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    product_payload = {
        **product_data,
        "image_folder": str(images_dir),
        "video_path": str(copied_video or ""),
        "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    _write_json(product_json, product_payload)
    _write_json(product_data_json, product_data)
    _write_csv(product_csv, product_payload, str(images_dir), str(copied_video or ""))
    _write_json(approval_json, approval_payload)
    _zip_export(export_path, zip_path)

    return {
        "export_dir": str(export_path),
        "product_json": str(product_json),
        "product_data_json": str(product_data_json),
        "product_csv": str(product_csv),
        "seller_approval_json": str(approval_json),
        "zip_path": str(zip_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exporter une fiche produit WAHA en JSON/CSV/ZIP.")
    parser.add_argument("product_data", help="Chemin vers product_data.json")
    parser.add_argument("--images", nargs="+", required=True, help="Images finales")
    parser.add_argument("--video", default="", help="Video produit")
    parser.add_argument("--output", default="output/export_manual", help="Dossier export")
    args = parser.parse_args()

    data = json.loads(Path(args.product_data).read_text(encoding="utf-8"))
    print(
        json.dumps(
            export_marketplace_package(data, args.images, args.video or None, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )
