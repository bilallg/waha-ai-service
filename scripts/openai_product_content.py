from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str | Path | None = None, *args: Any, **kwargs: Any) -> bool:
        env_path = Path(dotenv_path or ".env")
        if not env_path.exists():
            return False
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
DEFAULT_OPENAI_MODEL = "gpt-5.5-mini"

FORBIDDEN_TITLE_PATTERNS = (
    r"\bEAN\s*13\b",
    r"\bEAN\s*[-/]\s*13\b",
    r"\bEAN13\b",
    r"\bcode[-\s]?barres?\b",
    r"\bbarcode\b",
    r"\bproduit d[ée]tect[ée]\b",
    r"\bis a product identified with\b",
    r"\bfrom available product\s+data\b",
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _first_clean(product_data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _clean(product_data.get(key))
        if value:
            return value
    return ""


def _strip_forbidden_text(text: str, barcode: str = "") -> str:
    cleaned = _clean(text)
    for pattern in FORBIDDEN_TITLE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    if barcode:
        cleaned = re.sub(rf"\b{re.escape(barcode)}\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(?:price|prix)\s*:?\s*[$€£]?\s*\d+(?:[.,]\d{1,2})?\b", "", cleaned)
    cleaned = re.sub(r"(?i)\ba reduced risk of noncommunicable\s+chronic\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(?:common name|categories|brands|packaging)\s*:", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:;,.")
    return cleaned


def _normalize_quantity(text: str) -> str:
    return re.sub(r"\b(\d+(?:[.,]\d+)?)\s*(ml|cl|l|g|kg|mg)\b", r"\1 \2", text, flags=re.IGNORECASE)


def _limit_to_two_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", _clean(text))
    clean_sentences = [sentence for sentence in sentences if sentence]
    return " ".join(clean_sentences[:2]).strip()


def _title_is_generic(title: str, barcode: str = "") -> bool:
    cleaned = _strip_forbidden_text(title, barcode).lower()
    return not cleaned or cleaned in {"produit", "product", "description produit web non disponible"}


def _dedupe_parts(parts: list[str]) -> list[str]:
    selected: list[str] = []
    selected_lower = ""
    for part in parts:
        cleaned = _strip_forbidden_text(part)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in selected_lower:
            continue
        selected.append(cleaned)
        selected_lower = " ".join(selected).lower()
    return selected


def _useful_openai_metadata(product_data: dict[str, Any]) -> dict[str, str]:
    barcode = _clean(product_data.get("barcode"))
    raw_description = _first_clean(
        product_data,
        (
            "raw_description",
            "short_raw_description",
            "old_description",
            "original_description",
            "description",
            "product_description",
        ),
    )
    if re.search(r"(?i)\b(?:identified with|available product|common name|categories|brands|packaging|barcode)\b", raw_description):
        raw_description = ""
    metadata = {
        "brand": _first_clean(product_data, ("brand", "brands", "marque")),
        "product_name": _first_clean(product_data, ("product_name", "name", "title", "product")),
        "quantity": _first_clean(product_data, ("quantity", "size", "format", "contenance")),
        "packaging": _first_clean(product_data, ("packaging", "package", "packaging_text")),
        "flavor": _first_clean(product_data, ("flavor", "flavour", "flavors", "flavours", "taste")),
        "product_title": _first_clean(product_data, ("product_title", "source_product_title")),
        "raw_description": raw_description,
    }
    return {
        key: _normalize_quantity(_strip_forbidden_text(value, barcode))
        for key, value in metadata.items()
        if _normalize_quantity(_strip_forbidden_text(value, barcode))
    }


def _fallback_title(product_data: dict[str, Any], metadata: dict[str, str]) -> str:
    barcode = _clean(product_data.get("barcode"))
    parts = _dedupe_parts(
        [
            metadata.get("brand", ""),
            metadata.get("product_name", ""),
            metadata.get("quantity", ""),
        ]
    )
    if parts:
        return " ".join(parts)

    source_title = metadata.get("product_title") or _first_clean(product_data, ("title",))
    if source_title and not _title_is_generic(source_title, barcode):
        return _strip_forbidden_text(source_title, barcode)

    return "Produit"


def _fallback_content(product_data: dict[str, Any], metadata: dict[str, str]) -> dict[str, str]:
    product_title = _fallback_title(product_data, metadata)
    return {
        "product_title": product_title,
        "product_description": (
            f"{product_title} est un produit adapté à la vente en magasin, "
            "snack ou point de vente."
        ),
    }


def _validate_content(content: dict[str, Any], product_data: dict[str, Any]) -> dict[str, str]:
    barcode = _clean(product_data.get("barcode"))
    product_title = _strip_forbidden_text(content.get("product_title", ""), barcode)
    product_description = _limit_to_two_sentences(_strip_forbidden_text(content.get("product_description", ""), barcode))
    if not product_title or not product_description:
        raise ValueError("OpenAI response missing product_title or product_description")
    return {
        "product_title": product_title,
        "product_description": product_description,
    }


def generate_openai_product_content(product_data: dict[str, Any]) -> dict[str, str]:
    """Generate clean French marketplace title/description, with a local fallback."""
    metadata = _useful_openai_metadata(product_data)
    fallback = _fallback_content(product_data, metadata)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    LOGGER.info("OPENAI_API_KEY present: %s", bool(api_key))
    if not api_key:
        LOGGER.warning("OPENAI DESCRIPTION FAILED: OPENAI_API_KEY missing")
        LOGGER.info("FINAL DESCRIPTION SOURCE: clean_fallback")
        LOGGER.info("Final product_title: %s", fallback["product_title"])
        LOGGER.info("Final product_description: %s", fallback["product_description"])
        return fallback

    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    try:
        from openai import OpenAI

        LOGGER.info("CALLING OPENAI PRODUCT DESCRIPTION")
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=(
                "Tu es un rédacteur marketplace francophone. Réponds uniquement avec un objet JSON valide, "
                "sans Markdown et sans texte autour. Format exact: "
                "{\"product_title\":\"MONT BLANC Riz au lait vanille 570 g\","
                "\"product_description\":\"MONT BLANC Riz au lait vanille 570 g est un dessert lacté onctueux au goût vanille. "
                "Il convient aux rayons frais, épiceries, snacks et points de vente alimentaires.\"} "
                "Règles: français; style marketplace professionnel; titre au format Marque + nom produit + format; "
                "description maximum 2 phrases; aucun numéro de code-barres; aucune mention EAN13; "
                "aucune métadonnée technique; aucune catégorie; aucun prix; aucune allégation santé; "
                "aucune allégation santé générique, en français ou en anglais; "
                "ne pas inventer d'information absente des données."
            ),
            input=json.dumps(metadata, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "product_marketplace_content",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "product_title": {"type": "string"},
                            "product_description": {"type": "string"},
                        },
                        "required": ["product_title", "product_description"],
                    },
                }
            },
        )
        content = json.loads(response.output_text)
        generated = _validate_content(content, product_data)
        LOGGER.info("OPENAI DESCRIPTION SUCCESS")
        LOGGER.info("FINAL DESCRIPTION SOURCE: OpenAI")
        LOGGER.info("Final product_title: %s", generated["product_title"])
        LOGGER.info("Final product_description: %s", generated["product_description"])
        return generated
    except Exception as exc:
        LOGGER.warning("OPENAI DESCRIPTION FAILED: %s", exc, exc_info=True)
        LOGGER.info("FINAL DESCRIPTION SOURCE: clean_fallback")
        LOGGER.info("Final product_title: %s", fallback["product_title"])
        LOGGER.info("Final product_description: %s", fallback["product_description"])
        return fallback
