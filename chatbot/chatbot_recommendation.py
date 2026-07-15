from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def find_latest_product_data() -> dict[str, Any]:
    candidates = sorted(
        (PROJECT_ROOT / "output" / "exports").glob("product_*/product_data.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return _load_json(candidates[0]) if candidates else {}


def normalize_product_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
    if context and context.get("product_data"):
        return context["product_data"]
    if context and context.get("metadata"):
        metadata = context["metadata"]
        return {
            "barcode": metadata.get("barcode", context.get("barcode", "")),
            "title": metadata.get("title", ""),
            "brand": metadata.get("brand", ""),
            "category": metadata.get("category", ""),
            "marketplace_category": metadata.get("category", ""),
            "short_description": metadata.get("description", ""),
            "tags": [],
            "seo_keywords": [],
        }
    return find_latest_product_data()


def answer_seller_question(question: str, context: dict[str, Any] | None = None) -> str:
    product = normalize_product_context(context)
    query = question.lower().strip()
    title = product.get("seo_title") or product.get("title") or "ce produit"
    category = product.get("marketplace_category") or product.get("category") or "Categorie provisoire"
    tags = product.get("tags") or product.get("seo_keywords") or []
    brand = product.get("brand", "")

    if not query:
        return "Pose une question sur la categorie, le titre, le prix, les tags, les images ou les produits similaires."

    if any(word in query for word in ["categorie", "catégorie", "classer", "rubrique"]):
        return f"Choisis la categorie `{category}`. Verifie seulement que cette categorie existe dans ton back-office marketplace."

    if any(word in query for word in ["titre", "seo", "nom"]):
        brand_part = f"{brand} " if brand and brand.lower() not in title.lower() else ""
        return (
            f"Titre conseille: {brand_part}{title}. "
            "Garde la marque, le nom produit, le format et un mot categorie. Evite les promesses non verifiees."
        )

    if any(word in query for word in ["prix", "tarif", "dh", "marge"]):
        return (
            "Prix conseille: compare 3 a 5 fiches concurrentes de la meme categorie, puis place ton prix "
            "entre le prix median et le median +10% si tes images sont meilleures. Ajoute cout achat, livraison, commission et marge avant validation."
        )

    if any(word in query for word in ["tag", "mot-cle", "mot clé", "keyword"]):
        tag_text = ", ".join(tags[:10]) if tags else "produit, marketplace, vente-en-ligne"
        return f"Tags recommandes: {tag_text}. Garde 6 a 10 tags courts, sans repetition."

    if any(word in query for word in ["image", "photo", "visuel", "garder"]):
        return (
            "Garde les photos nettes, centrees, sur fond blanc, sans code-barres visible ni watermark. "
            "Priorite: face avant, angle secondaire, detail packaging, puis variante si elle existe."
        )

    if any(word in query for word in ["similaire", "complementaire", "vendre", "cross"]):
        return (
            f"Produits similaires a vendre: autres articles de `{category}`, formats voisins, packs de la meme marque "
            "et accessoires ou consommables compatibles si la categorie s'y prete."
        )

    if product:
        return (
            f"Pour {title}, commence par verifier la categorie `{category}`, garder 3 a 6 images propres, "
            "puis ajuster prix et stock avant publication."
        )

    return "Aucune fiche produit chargee. Lance le pipeline ou genere `output/exports/product_CODEBARRE/product_data.json`."


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="WAHA Chatbot", layout="wide")
    st.title("WAHA Chatbot vendeur")
    st.caption("Assistant local base sur la derniere fiche `product_data.json` generee.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Je peux aider pour categorie, titre, prix, tags, images et produits similaires."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Exemple: quels tags utiliser ?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = answer_seller_question(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
