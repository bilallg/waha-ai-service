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
COMPANY_OWNER_PATTERNS = (
    r"\bThe Coca[-\s]?Cola Company\b",
    r"\bCoca[-\s]?Cola Company\b",
    r"\bCompany\b",
    r"\bLLC\b",
    r"\bLtd\.?\b",
    r"\bInc\.?\b",
)
UNCERTAIN_TITLE_CLAIMS = {"low", "sugar", "zero", "bio", "light", "reduced"}
DESSERT_CONTAINER_WORDS = {
    "boite",
    "boîte",
    "bouteille",
    "canette",
    "creme",
    "crème",
    "dessert",
    "pack",
    "pot",
}
LOWERCASE_TITLE_WORDS = {"au", "aux", "de", "du", "des", "la", "le", "les", "vanille", "chocolat", "fraise"}
BAD_DESCRIPTION_PATTERNS = (
    r"\bdescription\s+fournie\b",
    r"\braw\s+metadata\b",
    r"\bmetadata\b",
    r"\bbarcode\b",
    r"\bEAN\s*13\b",
    r"\bEAN13\b",
    r"\bcategories?\b",
    r"\bpackaging\b",
    r"\bcommon\s+name\b",
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


def _looks_like_sprite_drink(text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:sprite|soft\s*drink|soda|boisson|gazeuse|can|canette|bottle|bouteille)\b", text or ""))


def normalize_quantity(text: str, product_data: dict[str, Any] | None = None) -> str:
    context = " ".join(
        str(value or "")
        for value in [
            text,
            *((product_data or {}).get(key, "") for key in ("barcode", "brand", "title", "name", "product_name", "product_title", "quantity", "packaging")),
        ]
    )
    normalized = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*(ml|cl|l|g|kg|mg)\b", r"\1 \2", text, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b(\d+(?:[.,]\d+)?)\s*(ml|cl|l|g|kg|mg)\b",
        lambda match: f"{match.group(1)} {match.group(2).lower()}",
        normalized,
        flags=re.IGNORECASE,
    )
    if _looks_like_sprite_drink(context):
        normalized = re.sub(r"(?i)\b330\s*cl\b", "330 ml", normalized)
    return normalized


def _normalize_quantity(text: str) -> str:
    return normalize_quantity(text)


def _strip_company_owner_text(text: str) -> str:
    cleaned = _clean(text)
    for pattern in COMPANY_OWNER_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _title_text_for_tokens(text: str, barcode: str = "", remove_sizes: bool = False, product_data: dict[str, Any] | None = None) -> str:
    cleaned = normalize_quantity(_strip_forbidden_text(text, barcode), product_data)
    cleaned = _strip_company_owner_text(cleaned)
    cleaned = re.sub(r"[–—_/|]+", " ", cleaned)
    cleaned = re.sub(r"[,\(\)\[\]]+", " ", cleaned)
    if remove_sizes:
        cleaned = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:ml|cl|l|g|kg|mg)\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _ascii_key(text: str) -> str:
    replacements = str.maketrans({"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "î": "i", "ï": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c"})
    return text.lower().translate(replacements)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?", text)


def _size_tokens(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:ml|cl|l|g|kg|mg)\b", _normalize_quantity(text), flags=re.IGNORECASE)


def _preferred_size(sizes: list[str], product_data: dict[str, Any] | None = None) -> str:
    if not sizes:
        return ""
    normalized = [normalize_quantity(size, product_data).lower() for size in sizes]
    ml_sizes = [size for size in normalized if size.endswith(" ml")]
    return ml_sizes[0] if ml_sizes else normalized[0]


def clean_product_title(raw_title: str, product_data: dict) -> str:
    """Return a concise Brand + product/flavor + size marketplace title."""
    if not isinstance(product_data, dict):
        product_data = {}
    barcode = _clean(product_data.get("barcode"))
    text = _title_text_for_tokens(raw_title, barcode, product_data=product_data)

    sizes_text = normalize_quantity(" ".join([text, _clean(product_data.get("quantity"))]), product_data)
    sizes = _size_tokens(sizes_text)
    preferred_size = _preferred_size(sizes, product_data)
    text_without_sizes = _title_text_for_tokens(text, barcode, remove_sizes=True, product_data=product_data)
    tokens = _word_tokens(text_without_sizes)

    brand = _title_text_for_tokens(product_data.get("brand", ""), barcode, remove_sizes=True)
    product_name = _title_text_for_tokens(
        product_data.get("product_name") or product_data.get("title") or product_data.get("name"),
        barcode,
        remove_sizes=True,
        product_data=product_data,
    )
    brand_tokens = _word_tokens(brand)
    product_tokens = _word_tokens(product_name)
    allowed_claims = {
        _ascii_key(token)
        for token in _word_tokens(_title_text_for_tokens(product_data.get("official_product_name", ""), barcode))
    }

    if not brand_tokens and tokens:
        brand_tokens = [tokens[0]]

    selected: list[str] = []
    seen: set[str] = set()

    def add_token(token: str) -> None:
        key = _ascii_key(token)
        if not key or key in seen:
            return
        if key == "sprite":
            token = "Sprite"
        selected.append(token)
        seen.add(key)

    for token in brand_tokens:
        add_token(token)

    candidate_tokens = product_tokens or tokens
    for token in candidate_tokens:
        key = _ascii_key(token)
        if key in seen or key in DESSERT_CONTAINER_WORDS:
            continue
        if key in UNCERTAIN_TITLE_CLAIMS and key not in allowed_claims:
            continue
        if key in LOWERCASE_TITLE_WORDS:
            token = token.lower()
        add_token(token)

    if brand_tokens and _ascii_key(" ".join(selected)) == _ascii_key(" ".join(brand_tokens)):
        for token in tokens:
            key = _ascii_key(token)
            if key in seen or key in DESSERT_CONTAINER_WORDS:
                continue
            if key in UNCERTAIN_TITLE_CLAIMS and key not in allowed_claims:
                continue
            if key in LOWERCASE_TITLE_WORDS:
                token = token.lower()
            add_token(token)

    cleaned_title = " ".join(selected).strip()
    if preferred_size:
        cleaned_title = f"{cleaned_title} {preferred_size}".strip()
    if _looks_like_sprite_drink(" ".join([cleaned_title, text, str(product_data)])):
        cleaned_title = re.sub(r"(?i)\b330\s*cl\b", "330 ml", cleaned_title)
    cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip(" -|:;,.")
    return cleaned_title or _normalize_quantity(text).strip() or "Produit"


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


def _has_repeated_leading_brand(text: str) -> bool:
    tokens = [_ascii_key(token) for token in _word_tokens(text)]
    return len(tokens) >= 2 and tokens[0] == tokens[1]


def _bad_generated_description(text: str, product_title: str, barcode: str = "") -> bool:
    cleaned = _clean(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in BAD_DESCRIPTION_PATTERNS):
        return True
    if barcode and barcode in cleaned:
        return True
    if _has_repeated_leading_brand(cleaned):
        return True
    title_key = _ascii_key(product_title)
    if "sprite" in title_key and not re.search(r"(?i)\b(?:boisson|gazeuse|rafra[iî]chissante|citron|lime|snacks|caf[ée]s?)\b", cleaned):
        return True
    return False


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
        key: normalize_quantity(_strip_forbidden_text(value, barcode), product_data)
        for key, value in metadata.items()
        if normalize_quantity(_strip_forbidden_text(value, barcode), product_data)
    }


def _fallback_title(product_data: dict[str, Any], metadata: dict[str, str]) -> str:
    if not isinstance(product_data, dict):
        product_data = {}
    barcode = _clean(product_data.get("barcode"))
    title_context = {**product_data, **metadata}
    parts = _dedupe_parts(
        [
            metadata.get("brand", ""),
            metadata.get("product_name", ""),
            metadata.get("quantity", ""),
        ]
    )
    if parts:
        return clean_product_title(" ".join(parts), title_context)

    source_title = metadata.get("product_title") or _first_clean(product_data, ("title",))
    if source_title and not _title_is_generic(source_title, barcode):
        return clean_product_title(source_title, title_context)

    return "Produit"


def _fallback_content(product_data: dict[str, Any], metadata: dict[str, str]) -> dict[str, str]:
    if not isinstance(product_data, dict):
        product_data = {}
    product_title = clean_product_title(_fallback_title(product_data, metadata), {**product_data, **metadata})
    title_key = _ascii_key(product_title)
    if "sprite" in title_key:
        return {
            "product_title": product_title,
            "product_description": (
                f"{product_title} est une boisson gazeuse rafraîchissante au goût citron-citron vert. "
                "Elle convient aux snacks, cafés, restaurants et points de vente."
            ),
        }
    return {
        "product_title": product_title,
        "product_description": (
            f"{product_title} est un produit adapté à la vente en magasin, "
            "snack ou point de vente."
        ),
    }


def clean_product_description(product_title: str, product_description: str, product_data: dict[str, Any] | None = None) -> str:
    product_data = product_data if isinstance(product_data, dict) else {}
    title = clean_product_title(product_title, product_data)
    barcode = _clean(product_data.get("barcode"))
    cleaned = _limit_to_two_sentences(normalize_quantity(_strip_forbidden_text(product_description, barcode), {**product_data, "product_title": title}))
    if _bad_generated_description(cleaned, title, barcode):
        return _fallback_content({**product_data, "product_title": title}, {"product_title": title})["product_description"]
    if cleaned and not re.search(r"[.!?]$", cleaned):
        cleaned = f"{cleaned}."
    return cleaned


def _validate_content(content: dict[str, Any], product_data: dict[str, Any]) -> dict[str, str]:
    if not isinstance(product_data, dict):
        product_data = {}
    if not isinstance(content, dict):
        content = {}
    barcode = _clean(product_data.get("barcode"))
    product_title = clean_product_title(content.get("product_title", ""), product_data)
    product_description = _limit_to_two_sentences(_strip_forbidden_text(content.get("product_description", ""), barcode))
    if not product_title or not product_description:
        raise ValueError("OpenAI response missing product_title or product_description")
    if _bad_generated_description(product_description, product_title, barcode):
        raise ValueError("OpenAI response contains raw metadata or low-quality description")
    return {
        "product_title": product_title,
        "product_description": product_description,
    }


def generate_openai_product_content(product_data: dict[str, Any]) -> dict[str, str]:
    """Generate clean French marketplace title/description, with a local fallback."""
    if not isinstance(product_data, dict):
        product_data = {}
    metadata = _useful_openai_metadata(product_data)
    fallback = _fallback_content(product_data, metadata)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    LOGGER.info("OPENAI_API_KEY present: %s", bool(api_key))
    LOGGER.info("OPENAI_MODEL: %s", model)
    LOGGER.info("CALLING OPENAI PRODUCT DESCRIPTION")
    if not api_key:
        LOGGER.warning("OPENAI DESCRIPTION FAILED: OPENAI_API_KEY missing")
        LOGGER.info("FINAL DESCRIPTION SOURCE: clean_fallback")
        LOGGER.info("Final product_title: %s", fallback["product_title"])
        LOGGER.info("Final product_description: %s", fallback["product_description"])
        return {**fallback, "description_source": "clean_fallback"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=20.0)
        response = client.responses.create(
            model=model,
            instructions=(
                "Tu es un rédacteur marketplace francophone. Réponds uniquement avec un objet JSON valide, "
                "sans Markdown et sans texte autour. Format exact: "
                "{\"product_title\":\"Sprite 330 ml\","
                "\"product_description\":\"Sprite 330 ml est une boisson gazeuse rafraîchissante au goût citron-citron vert. "
                "Elle convient aux snacks, cafés, restaurants et points de vente.\"} "
                "Règles: français; style marketplace professionnel; titre au format Marque + nom produit + format; "
                "description maximum 2 phrases; aucun numéro de code-barres; aucune mention EAN13; "
                "aucune métadonnée technique; aucune catégorie; aucun prix; aucune allégation santé; "
                "ne jamais écrire 'Description fournie', 'catégories', 'packaging', 'barcode' ou du texte brut; "
                "ne pas répéter la marque; corriger 330 cl en 330 ml pour une canette Sprite; "
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
        return {**generated, "description_source": "OpenAI"}
    except Exception as exc:
        LOGGER.warning("OPENAI DESCRIPTION FAILED: %s", exc, exc_info=True)
        LOGGER.info("FINAL DESCRIPTION SOURCE: clean_fallback")
        LOGGER.info("Final product_title: %s", fallback["product_title"])
        LOGGER.info("Final product_description: %s", fallback["product_description"])
        return {**fallback, "description_source": "clean_fallback"}
