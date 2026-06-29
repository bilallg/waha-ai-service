from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _clean(value: object, fallback: str = "") -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def _slug_words(*values: str) -> list[str]:
    words: list[str] = []
    for value in values:
        for word in re.findall(r"[A-Za-zÀ-ÿ0-9]+", value.lower()):
            if len(word) >= 3 and word not in words:
                words.append(word)
    return words


def _marketplace_category(category: str, title: str) -> str:
    text = f"{category} {title}".lower()
    if any(word in text for word in ["boisson", "drink", "eau", "jus", "soda"]):
        return "Epicerie > Boissons"
    if any(word in text for word in ["chocolat", "biscuit", "snack", "bonbon"]):
        return "Epicerie > Snacks et confiserie"
    if any(word in text for word in ["savon", "shampoo", "gel", "douche", "beaute"]):
        return "Beaute et hygiene"
    if any(word in text for word in ["electronique", "phone", "chargeur", "cable"]):
        return "Electronique"
    if any(word in text for word in ["vetement", "chaussure", "sac", "mode"]):
        return "Mode"
    return category or "Categorie provisoire"


def generate_product_description(product: dict[str, Any]) -> dict[str, Any]:
    barcode = _clean(product.get("barcode"), "unknown")
    raw_title = _clean(product.get("title"), f"Produit {barcode}")
    brand = _clean(product.get("brand"))
    category = _clean(product.get("category"), "Categorie provisoire")
    quantity = _clean(product.get("quantity"))
    source_description = _clean(product.get("description"), "Description provisoire a valider par le vendeur")

    title_parts = [brand, raw_title, quantity]
    seo_title = " ".join(part for part in title_parts if part)
    if len(seo_title) > 95:
        seo_title = seo_title[:92].rstrip() + "..."

    marketplace_category = _marketplace_category(category, raw_title)
    short_description = (
        f"{seo_title} est pret pour une fiche marketplace claire, avec images produit verifiees et description vendeur."
    )
    if product.get("source") == "fallback":
        short_description = f"{seo_title}: fiche provisoire a completer et valider par le vendeur."

    long_description = (
        f"{source_description}. Cette fiche a ete generee automatiquement a partir du code-barres {barcode}. "
        "Les visuels finaux sont prepares sur fond blanc pour une presentation marketplace propre. "
        "Le vendeur doit verifier le prix, le stock, les variantes et les informations legales avant publication."
    )

    base_tags = _slug_words(raw_title, brand, category, quantity)
    tags = base_tags[:10]
    if "marketplace" not in tags:
        tags.append("marketplace")

    bullet_points = [
        f"Produit identifie par code-barres {barcode}",
        f"Categorie recommandee: {marketplace_category}",
        "Images finales optimisees pour fiche e-commerce",
        "Description et mots-cles generes automatiquement",
    ]
    if brand:
        bullet_points.insert(1, f"Marque detectee: {brand}")
    if quantity:
        bullet_points.insert(2, f"Format detecte: {quantity}")

    return {
        "barcode": barcode,
        "title": raw_title,
        "brand": brand,
        "category": category,
        "source": _clean(product.get("source"), "fallback"),
        "seo_title": seo_title,
        "short_description": short_description,
        "long_description": long_description,
        "bullet_points": bullet_points,
        "tags": tags,
        "marketplace_category": marketplace_category,
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
