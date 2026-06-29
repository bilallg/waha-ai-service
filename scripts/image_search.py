from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _unique_urls(urls: Iterable[str]) -> list[str]:
    clean_urls: list[str] = []
    seen: set[str] = set()

    for url in urls:
        if not url or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        seen.add(url)
        clean_urls.append(url)

    return clean_urls


def search_image_urls(
    barcode: str,
    max_results: int = 60,
    api_key: str | None = None,
    query: str | None = None,
) -> list[str]:
    api_key = api_key or os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("SERPAPI_API_KEY non defini. La recherche web est ignoree.")
        return []

    if GoogleSearch is None:
        print("Le package google-search-results n'est pas installe.")
        return []

    params = {
        "engine": "google",
        "q": query or barcode,
        "tbm": "isch",
        "ijn": "0",
        "api_key": api_key,
        "safe": "active",
    }

    try:
        results = GoogleSearch(params).get_dict()
    except Exception as exc:
        print(f"Erreur SerpAPI ignoree: {exc}")
        return []

    image_results = results.get("images_results", [])
    urls = []
    for item in image_results:
        urls.extend(
            [
                item.get("original"),
                item.get("thumbnail"),
            ]
        )

    return _unique_urls(urls)[:max_results]


def _save_response_as_jpeg(response: requests.Response, output_path: Path) -> bool:
    try:
        response.raw.decode_content = True
        with Image.open(response.raw) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.save(output_path, "JPEG", quality=92, optimize=True)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def download_image(url: str, output_path: Path, timeout: int = 12) -> bool:
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "image" not in content_type:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return _save_response_as_jpeg(response, output_path)
    except Exception as exc:
        print(f"Lien image ignore: {url} ({exc})")
        return False


def search_and_download(
    barcode: str,
    output_dir: str | Path,
    max_images: int = 30,
    api_key: str | None = None,
    query: str | None = None,
) -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    urls = search_image_urls(barcode=barcode, max_results=max_images * 3, api_key=api_key, query=query)
    saved: list[str] = []

    for index, url in enumerate(urls, start=1):
        if len(saved) >= max_images:
            break

        image_path = output_path / f"download_{len(saved) + 1}.jpg"
        if download_image(url, image_path):
            saved.append(str(image_path))
            print(f"Image telechargee: {image_path}")
        else:
            print(f"Image ignoree #{index}")

    print(f"Total images telechargees: {len(saved)}")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recherche et telechargement d'images via SerpAPI.")
    parser.add_argument("barcode", help="Code-barres utilise comme requete")
    parser.add_argument("--output", default="main/web_images/manual_test", help="Dossier de sortie")
    parser.add_argument("--max-images", type=int, default=30, help="Nombre maximal d'images")
    args = parser.parse_args()

    search_and_download(args.barcode, args.output, args.max_images)
