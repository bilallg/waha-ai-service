from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FALLBACK_DESCRIPTION = "Product details are limited and can be completed manually by the seller before publication."


def _clean(value: object, fallback: str = "") -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def _clean_sentence_text(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:click here|read more(?:\s+at)?|see more|visit site|shop now|buy now)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?i)\b(?:price|prix|cost|sale)\s*:?\s*(?:[$€£]|mad|dh|dhs|usd|eur|gbp)?\s*\d*(?:[.,]\d{1,2})?",
        " ",
        text,
    )
    text = re.sub(r"(?i)(?:[$€£]\s*\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s*(?:mad|dh|dhs|usd|eur|gbp))", " ", text)
    text = re.sub(r"\b(?:google search|google images|image result|search result|serpapi)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,.")
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return text


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences: list[str] = []
    seen: set[str] = set()
    for sentence in raw_sentences:
        cleaned = _clean_sentence_text(sentence).strip()
        if not cleaned or len(cleaned) < 18:
            continue
        if not re.search(r"[A-Za-z]{3,}", cleaned):
            continue
        if re.search(
            r"\b(?:might|could|probably|uncertain|verify before validation|validation|a verifier|verifier|vérifier|pourrait|"
            r"produit detecte|produit détecté|resultats web|résultats web|meilleur resultat web|"
            r"meilleur résultat web|description generee automatiquement|description générée automatiquement)\b",
            cleaned,
            re.IGNORECASE,
        ):
            continue
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(cleaned)
        if len(sentences) >= 2:
            break
    return sentences


def _has_specific_title(title: str, barcode: str) -> bool:
    cleaned = _clean_sentence_text(title)
    if not cleaned or cleaned.lower() in {"product", "produit"}:
        return False
    if barcode and cleaned in {barcode, f"Produit {barcode}", f"Product {barcode}", f"Produit détecté - {barcode}"}:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", cleaned))


def generate_clean_product_description(
    barcode: str = "",
    barcode_type: str = "",
    product_title: str = "",
    old_description: str = "",
    links: list[dict[str, Any]] | None = None,
    source: str = "",
) -> str:
    """Create a concise e-commerce description from barcode enrichment data."""
    cleaned_barcode = _clean(barcode)
    cleaned_title = _clean_sentence_text(product_title)
    old_sentences = _split_sentences(old_description)

    if not _has_specific_title(cleaned_title, cleaned_barcode) and not old_sentences:
        return FALLBACK_DESCRIPTION

    sentences: list[str] = []
    if _has_specific_title(cleaned_title, cleaned_barcode):
        sentences.extend(_title_based_description(cleaned_title))

    title_key = cleaned_title.lower().strip(" .!?")
    related_old_sentences = (
        sentence
        for sentence in old_sentences
        if sentence.lower().strip(" .!?") != title_key
        and _is_product_related_sentence(sentence, cleaned_title)
    )
    sentences.extend(related_old_sentences)

    if len(sentences) < 2:
        sentences.append("It can be listed with seller-reviewed packaging details in the platform.")

    cleaned_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        cleaned = _clean_sentence_text(sentence)
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_sentences.append(cleaned)
        if len(cleaned_sentences) >= 3:
            break

    if not cleaned_sentences:
        return FALLBACK_DESCRIPTION

    description = " ".join(cleaned_sentences[:3])
    if cleaned_barcode:
        description = re.sub(rf"\b{re.escape(cleaned_barcode)}\b", "", description)
        description = re.sub(r"\s+", " ", description).strip()
    if len(description) > 700:
        description = description[:697].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return description or FALLBACK_DESCRIPTION


def _title_based_description(title: str) -> list[str]:
    lowered = title.lower()
    if "sprite" in lowered:
        return [
            f"{title} is a lemon-lime flavoured soft drink.",
            "It is commonly served chilled and is suitable for retail stores, snack menus, restaurants and daily refreshment.",
        ]
    if "coca-cola" in lowered or "coca cola" in lowered or re.search(r"\bcoke\b", lowered):
        return [
            f"{title} is a cola soft drink.",
            "It is commonly served chilled and can be listed as a ready-to-sell packaged drink.",
        ]
    if "fanta" in lowered:
        return [
            f"{title} is a fruit-flavoured carbonated soft drink.",
            "It is commonly served chilled and can be listed as a ready-to-sell packaged drink.",
        ]
    if "pepsi" in lowered:
        return [
            f"{title} is a cola soft drink.",
            "It is commonly served chilled and can be listed as a ready-to-sell packaged drink.",
        ]
    if any(word in lowered for word in ["chocolate", "chocolat", "milka"]):
        return [
            f"{title} is a packaged chocolate product.",
            "It can be listed with seller-reviewed packaging details for retail sale.",
        ]
    if any(word in lowered for word in ["water", "eau"]):
        return [
            f"{title} is a packaged water product.",
            "It can be listed with seller-reviewed packaging details for retail sale.",
        ]
    return [
        f"{title} is a packaged consumer product.",
        "It can be listed with seller-reviewed packaging details in the platform.",
    ]


def generate_clean_product_description_from_payload(payload: dict[str, Any]) -> str:
    return generate_clean_product_description(
        barcode=_clean(payload.get("barcode")),
        barcode_type=_clean(payload.get("barcode_type")),
        product_title=_clean(payload.get("product_title") or payload.get("title")),
        old_description=_clean(payload.get("old_description") or payload.get("original_description") or payload.get("description")),
        links=payload.get("links") or [],
        source=_clean(payload.get("source")),
    )


def _slug_words(*values: str) -> list[str]:
    words: list[str] = []
    for value in values:
        for word in re.findall(r"[A-Za-zÀ-ÿ0-9]+", value.lower()):
            if len(word) >= 3 and word not in words:
                words.append(word)
    return words


def _title_tokens(title: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[A-Za-zÀ-ÿ0-9]+", title.lower())
        if len(word) >= 3
    }


def _is_product_related_sentence(sentence: str, title: str) -> bool:
    title_words = _title_tokens(title)
    if not title_words:
        return False
    sentence_words = _title_tokens(sentence)
    if not sentence_words.intersection(title_words):
        return False
    unrelated_markers = [
        "search",
        "result",
        "image",
        "review",
        "blog",
        "news",
        "compare",
        "coupon",
        "price",
        "shipping",
        "delivery",
        "nutrition",
        "protein",
        "proteins",
        "calcium",
        "iron",
        "vitamin",
        "vitamins",
        "health",
        "benefit",
        "benefits",
        "calorie",
        "calories",
    ]
    lowered = sentence.lower()
    return not any(marker in lowered for marker in unrelated_markers)


def generate_product_description(product: dict[str, Any]) -> dict[str, Any]:
    barcode = _clean(product.get("barcode"), "unknown")
    raw_title = _clean(product.get("title"), f"Produit {barcode}")
    brand = _clean(product.get("brand"))
    quantity = _clean(product.get("quantity"))
    source_description = _clean_sentence_text(product.get("description")) or "Description provisoire a valider par le vendeur"
    clean_description = generate_clean_product_description(
        barcode=barcode,
        barcode_type=_clean(product.get("barcode_type")),
        product_title=raw_title,
        old_description=source_description,
        links=product.get("links") or [],
        source=_clean(product.get("source")),
    )

    title_parts = [brand, raw_title, quantity]
    seo_title = " ".join(part for part in title_parts if part)
    if len(seo_title) > 95:
        seo_title = seo_title[:92].rstrip() + "..."

    short_description = (
        f"{seo_title} is ready for a seller-reviewed product listing."
    )
    if product.get("source") == "fallback":
        short_description = f"{seo_title}: product details should be completed and reviewed by the seller."

    long_description = (
        f"{clean_description} The seller should review the physical product details before publication."
    )

    base_tags = _slug_words(raw_title, brand, quantity)
    tags = base_tags[:10]
    if "marketplace" not in tags:
        tags.append("marketplace")

    bullet_points = [
        "Product information prepared for seller review",
        "Images finales optimisees pour fiche e-commerce",
        "Description generated for seller review",
    ]
    if brand:
        bullet_points.insert(1, f"Marque detectee: {brand}")
    if quantity:
        bullet_points.insert(2, f"Format detecte: {quantity}")

    return {
        "barcode": barcode,
        "title": raw_title,
        "brand": brand,
        "source": _clean(product.get("source"), "fallback"),
        "product_description": clean_description,
        "expiration_date": product.get("expiration_date"),
        "expiration_found": bool(product.get("expiration_found")),
        "seo_title": seo_title,
        "short_description": short_description,
        "long_description": long_description,
        "bullet_points": bullet_points,
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
