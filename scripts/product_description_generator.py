from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .openai_product_content import generate_openai_product_content
except ImportError:
    from openai_product_content import generate_openai_product_content


def _clean(value: object, fallback: str = "") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or fallback


def _clean_title(title: str, barcode: str = "") -> str:
    cleaned = _clean(title)
    if barcode:
        cleaned = re.sub(rf"\b{re.escape(barcode)}\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(?:EAN\s*13|EAN13|barcode|code[-\s]?barres?)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:;,.")
    return cleaned or "Produit"


def _fallback_description(title: str) -> str:
    return f"{title} est un produit alimentaire adapté à la vente en magasin, snack ou point de vente."


def _slug_words(*values: str) -> list[str]:
    words: list[str] = []
    for value in values:
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", str(value or "").lower()):
            if len(word) >= 3 and word not in words:
                words.append(word)
            if len(words) >= 10:
                return words
    return words


def generate_clean_product_description(
    barcode: str = "",
    barcode_type: str = "",
    product_title: str = "",
    old_description: str = "",
    links: list[dict[str, Any]] | None = None,
    source: str = "",
) -> str:
    title = _clean_title(product_title, barcode)
    return _fallback_description(title)


def generate_clean_product_description_from_payload(payload: dict[str, Any]) -> str:
    return generate_clean_product_description(
        barcode=_clean(payload.get("barcode")),
        barcode_type=_clean(payload.get("barcode_type")),
        product_title=_clean(payload.get("product_title") or payload.get("title")),
        old_description=_clean(payload.get("old_description") or payload.get("original_description") or payload.get("description")),
        links=payload.get("links") or [],
        source=_clean(payload.get("source")),
    )


def generate_product_description(product: dict[str, Any]) -> dict[str, Any]:
    content = generate_openai_product_content(product)
    title = _clean_title(content.get("product_title") or product.get("title"), _clean(product.get("barcode")))
    description = _clean(content.get("product_description"), _fallback_description(title))
    brand = _clean(product.get("brand"))
    quantity = _clean(product.get("quantity"))
    tags = _slug_words(title, brand, quantity)
    if "marketplace" not in tags:
        tags.append("marketplace")

    return {
        "barcode": _clean(product.get("barcode")),
        "title": title,
        "brand": brand,
        "source": _clean(product.get("source"), "fallback"),
        "product_description": description,
        "expiration_date": product.get("expiration_date"),
        "expiration_found": bool(product.get("expiration_found")),
        "seo_title": title,
        "short_description": description,
        "long_description": description,
        "bullet_points": [],
        "tags": tags,
        "seo_keywords": tags[:],
        "quantity": quantity,
        "image_url": _clean(product.get("image_url")),
    }


def save_product_data(product: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    data = generate_product_description(product)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generer product_data.json pour WAHA.")
    parser.add_argument("product_json", help="Fichier JSON contenant les donnees produit lookup.")
    parser.add_argument("--output", default="product_data.json", help="Chemin de sortie")
    args = parser.parse_args()

    product_payload = json.loads(Path(args.product_json).read_text(encoding="utf-8"))
    print(json.dumps(save_product_data(product_payload, args.output), indent=2, ensure_ascii=False))
