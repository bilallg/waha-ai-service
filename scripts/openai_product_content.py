from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

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

try:
    from .product_description_generator import generate_clean_product_description
except ImportError:
    from product_description_generator import generate_clean_product_description


load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_FALLBACK_LOG = "OpenAI description generation failed, using fallback."

FORBIDDEN_TITLE_PATTERNS = (
    r"\bEAN\s*13\b",
    r"\bEAN13\b",
    r"\bcode[-\s]?barres?\b",
    r"\bbarcode\b",
    r"\bproduit d[ée]tect[ée]\b",
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
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:;,.")
    return cleaned


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
    metadata = {
        "brand": _first_clean(product_data, ("brand", "brands", "marque")),
        "product_name": _first_clean(product_data, ("product_name", "name", "title", "product")),
        "quantity": _first_clean(product_data, ("quantity", "size", "format", "contenance")),
        "packaging": _first_clean(product_data, ("packaging", "package", "packaging_text")),
        "flavor": _first_clean(product_data, ("flavor", "flavour", "flavors", "flavours", "taste")),
        "product_title": _first_clean(product_data, ("product_title", "source_product_title")),
        "raw_description": _first_clean(
            product_data,
            (
                "raw_description",
                "short_raw_description",
                "old_description",
                "original_description",
                "description",
                "product_description",
            ),
        ),
    }
    return {key: value for key, value in metadata.items() if value}


def _fallback_title(product_data: dict[str, Any], metadata: dict[str, str]) -> str:
    barcode = _clean(product_data.get("barcode"))
    source_title = metadata.get("product_title") or metadata.get("product_name") or _first_clean(product_data, ("title",))
    if source_title and not _title_is_generic(source_title, barcode):
        return _strip_forbidden_text(source_title, barcode)

    parts = _dedupe_parts(
        [
            metadata.get("brand", ""),
            metadata.get("product_name", ""),
            metadata.get("quantity", ""),
        ]
    )
    return " ".join(parts) or "Produit"


def _fallback_content(product_data: dict[str, Any], metadata: dict[str, str]) -> dict[str, str]:
    barcode = _clean(product_data.get("barcode"))
    product_title = _fallback_title(product_data, metadata)
    product_description = generate_clean_product_description(
        barcode=barcode,
        barcode_type=_clean(product_data.get("barcode_type")),
        product_title=product_title,
        old_description=metadata.get("raw_description", ""),
        links=product_data.get("links") or [],
        source=_clean(product_data.get("source")),
    )
    return {
        "product_title": product_title,
        "product_description": _limit_to_two_sentences(_strip_forbidden_text(product_description, barcode)),
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
    if not api_key:
        LOGGER.warning(OPENAI_FALLBACK_LOG)
        return fallback

    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=(
                "Tu es un redacteur marketplace francophone. Genere uniquement un JSON valide. "
                "Regles: langue francaise; titre propre au format Marque + nom produit + format; "
                "description courte et professionnelle en 2 phrases maximum; ne pas inclure le code-barres; "
                "ne pas mentionner EAN13; ne pas inclure categorie, prix ou metadata technique; "
                "ne pas inventer d'informations non presentes dans les donnees; si le produit est une boisson, "
                "mentionner le gout et un cas d'utilisation commercial."
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
        return _validate_content(content, product_data)
    except Exception:
        LOGGER.warning(OPENAI_FALLBACK_LOG, exc_info=True)
        return fallback
