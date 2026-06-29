from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

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


load_dotenv()

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
REQUEST_TIMEOUT = 15
RESULT_SOURCE = "SerpAPI"
UNCERTAIN_WARNING = "Description générée automatiquement à partir des résultats web. À vérifier avant validation."

PRODUCT_WORDS = {
    "aliment",
    "amazon",
    "beverage",
    "biscuit",
    "boisson",
    "brand",
    "can",
    "cereal",
    "chocolat",
    "cola",
    "coffee",
    "cosmetic",
    "drink",
    "eau",
    "flavored",
    "food",
    "fromage",
    "juice",
    "grocery",
    "lait",
    "lemon",
    "market",
    "ml",
    "pack",
    "product",
    "produit",
    "savon",
    "shampoo",
    "snack",
    "soda",
    "supermarket",
    "tea",
}

TRUSTED_PRODUCT_DOMAINS = {
    "openfoodfacts.org",
    "world.openfoodfacts.org",
    "fr.openfoodfacts.org",
    "upcitemdb.com",
    "go-upc.com",
    "barcodelookup.com",
}

ECOMMERCE_DOMAIN_HINTS = {
    "amazon.",
    "carrefour.",
    "walmart.",
    "target.",
    "kroger.",
    "tesco.",
    "sainsburys.",
    "asda.",
    "instacart.",
    "noon.",
    "jumia.",
    "ubuy.",
    "desertcart.",
    "marjane.",
    "aswakassalam.",
    "auchan.",
    "monoprix.",
    "leclerc.",
    "coles.",
    "woolworths.",
}

FMCG_CATALOG_DOMAIN_HINTS = {
    "openfoodfacts.",
    "upcitemdb.",
    "go-upc.",
    "barcodelookup.",
    "foodrepo.",
    "fooducate.",
    "myfitnesspal.",
    "nutritionvalue.",
    "packaging-labelling.",
}

PRODUCT_PATH_HINTS = (
    "/product",
    "/products",
    "/produit",
    "/item",
    "/p/",
    "/dp/",
    "/itm/",
    "/shop/",
)

ARTICLE_PATH_HINTS = (
    "/article",
    "/articles",
    "/blog",
    "/blogs",
    "/news",
    "/image",
    "/images",
    "/category",
    "/categories",
)

GENERIC_TITLE_PATTERNS = (
    r"\bbranded foods?\b",
    r"\bgeneric vs\b",
    r"\bname-brand foods?\b",
    r"\bfood brands?\b",
    r"\bbrand name foods?\b",
    r"\bfood brand collaborations?\b",
    r"\bbrand collaborations?\b",
    r"\bfmcg brands?\b",
    r"\btop food brands?\b",
    r"\bimages?\b",
    r"\barticle\b",
    r"\bblog\b",
    r"\bnews\b",
)

SOURCE_LABELS = (
    "Open Food Facts",
    "OpenFoodFacts",
    "UPCitemdb",
    "UPC Item DB",
    "Barcode Lookup",
    "Go-UPC",
    "Amazon",
    "Carrefour",
    "Walmart",
    "Target",
)

USELESS_TITLE_PARTS = [
    "Google Search",
    "Google Images",
    "Google",
    "Search",
    "Images",
    "Amazon.com",
]


def _empty_result(barcode: str, warning: str = "") -> dict[str, Any]:
    return {
        "success": False,
        "barcode": str(barcode or ""),
        "product_title": f"Produit détecté - {barcode}",
        "product_description": "",
        "images": [],
        "links": [],
        "source": RESULT_SOURCE,
        "warning": warning,
    }


def _api_key() -> str:
    return os.getenv("SERPAPI_API_KEY", "").strip()


def _request_serpapi(params: dict[str, Any]) -> dict[str, Any]:
    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY manquante. Le barcode est détecté, mais les informations produit/images web ne peuvent pas être récupérées."
        )

    response = requests.get(
        SERPAPI_ENDPOINT,
        params={**params, "api_key": api_key},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "WAHA-Platform/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    if payload.get("search_metadata", {}).get("status") == "Error":
        raise RuntimeError(str(payload.get("search_metadata", {}).get("error") or "Erreur SerpAPI"))

    return payload


def _clean_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _clean_title(title: str, barcode: str) -> str:
    cleaned = _clean_text(title)
    for useless in USELESS_TITLE_PARTS:
        cleaned = re.sub(rf"\b{re.escape(useless)}\b", "", cleaned, flags=re.IGNORECASE)

    for separator in [" | ", " - ", " – ", " — ", " : "]:
        if separator not in cleaned:
            continue
        parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
        if len(parts) < 2:
            continue
        trailing = parts[-1].lower()
        if any(label.lower() in trailing for label in SOURCE_LABELS) or re.search(r"\.(com|org|net|ma|fr|uk)$", trailing):
            cleaned = separator.join(parts[:-1])

    cleaned = re.sub(r"^(buy|acheter|shop)\s+", "", cleaned, flags=re.IGNORECASE)
    if barcode:
        without_barcode = re.sub(rf"\b{re.escape(barcode)}\b", "", cleaned)
        without_barcode = re.sub(r"\s+", " ", without_barcode).strip(" -|:")
        if re.search(r"[A-Za-zÀ-ÿ]{3,}", without_barcode):
            cleaned = without_barcode
    cleaned = re.sub(r"\s*[-|:]\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:")

    if barcode and cleaned.count(barcode) > 1:
        cleaned = cleaned.replace(barcode, "", cleaned.count(barcode) - 1)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:")

    return cleaned


def _looks_product_like(title: str) -> bool:
    lowered = title.lower()
    return any(word in lowered for word in PRODUCT_WORDS) or bool(re.search(r"\b\d+\s?(g|kg|ml|l|cl|oz)\b", lowered))


def _is_generic_title(title: str) -> bool:
    lowered = title.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in GENERIC_TITLE_PATTERNS)


def _domain_parts(link: str) -> tuple[str, str]:
    parsed = urlparse(link)
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    return domain, path


def _is_trusted_product_domain(domain: str) -> bool:
    return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in TRUSTED_PRODUCT_DOMAINS)


def _is_ecommerce_domain(domain: str) -> bool:
    return any(hint in domain for hint in ECOMMERCE_DOMAIN_HINTS)


def _is_fmcg_catalog_domain(domain: str) -> bool:
    return any(hint in domain for hint in FMCG_CATALOG_DOMAIN_HINTS)


def _has_product_path(path: str) -> bool:
    return any(hint in path for hint in PRODUCT_PATH_HINTS)


def _has_article_path(path: str) -> bool:
    return any(hint in path for hint in ARTICLE_PATH_HINTS)


def _contains_exact_barcode(item: dict[str, str], barcode: str) -> bool:
    title = _clean_title(item.get("title", ""), barcode)
    snippet = _clean_text(item.get("snippet"))
    return bool(barcode and str(barcode).lower() in f"{title} {snippet}".lower())


def _has_product_name_signal(title: str) -> bool:
    if _is_generic_title(title):
        return False
    has_letters = bool(re.search(r"[A-Za-zÀ-ÿ]{3,}", title))
    has_quantity = bool(re.search(r"\b\d+\s?(g|kg|mg|ml|l|cl|oz|can|pack)\b", title.lower()))
    has_multiple_words = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", title)) >= 2
    return has_letters and (has_quantity or has_multiple_words)


def _is_reliable_product_source(link: str) -> bool:
    domain, path = _domain_parts(link)
    return (
        _is_trusted_product_domain(domain)
        or _is_ecommerce_domain(domain)
        or _is_fmcg_catalog_domain(domain)
        or _has_product_path(path)
    )


def _is_selectable_product_result(item: dict[str, str], barcode: str) -> bool:
    title = _clean_title(item.get("title", ""), barcode)
    snippet = _clean_text(item.get("snippet"))
    if not title or _is_generic_title(title):
        return False
    return (
        _contains_exact_barcode(item, barcode)
        or _is_reliable_product_source(item.get("link", ""))
        or _looks_product_like(title)
        or _looks_product_like(snippet)
        or _has_product_name_signal(title)
    )


def _result_score(item: dict[str, str], barcode: str, index: int) -> tuple[int, int, int, int, int]:
    title = _clean_title(item.get("title", ""), barcode)
    snippet = _clean_text(item.get("snippet"))
    link = _clean_text(item.get("link"))
    combined = f"{title} {snippet}".lower()
    domain, path = _domain_parts(link)

    exact_barcode = str(barcode).lower() in combined
    trusted_domain = _is_trusted_product_domain(domain)
    ecommerce_product = _is_ecommerce_domain(domain) or _has_product_path(path)
    fmcg_catalog = _is_fmcg_catalog_domain(domain)
    article_path = _has_article_path(path)
    product_like = _looks_product_like(title) or _looks_product_like(snippet)
    generic = _is_generic_title(title)
    has_specific_name = _has_product_name_signal(title)

    score = 0
    if exact_barcode:
        score += 1000
    if trusted_domain:
        score += 450
    if ecommerce_product:
        score += 275
    if fmcg_catalog:
        score += 225
    if product_like:
        score += 180
    if has_specific_name:
        score += 160
    if re.search(r"\b\d+\s?(g|kg|ml|l|cl|oz)\b", title.lower()):
        score += 80
    if generic:
        score -= 1500
    if article_path and not ecommerce_product:
        score -= 350
    if len(title) < 8:
        score -= 100
    if len(title) > 95:
        score -= 60

    # Keep earlier SerpAPI results as a final tie-breaker without letting order beat evidence.
    order_bonus = max(0, 50 - index)
    compact_title_bonus = max(0, 120 - len(title))
    return score, order_bonus, compact_title_bonus, -len(title), -index


def _select_best_result(links: list[dict[str, str]], barcode: str) -> dict[str, str]:
    candidates = [
        {**item, "title": _clean_title(item.get("title", ""), barcode)}
        for item in links
        if _clean_title(item.get("title", ""), barcode)
    ]
    candidates = [item for item in candidates if item["title"] != barcode]
    if not candidates:
        return {
            "title": f"Produit détecté - {barcode}",
            "link": "",
            "snippet": "",
        }

    selectable_candidates = [item for item in candidates if _is_selectable_product_result(item, barcode)]
    scoring_candidates = selectable_candidates or candidates
    indexed_candidates = list(enumerate(scoring_candidates))
    return max(indexed_candidates, key=lambda pair: _result_score(pair[1], barcode, pair[0]))[1]


def _build_description(barcode: str, title: str, selected_result: dict[str, str], certain: bool) -> str:
    selected_snippet = _clean_text(selected_result.get("snippet"))

    if selected_snippet:
        description = (
            f"Produit détecté à partir du code-barres {barcode}. "
            f"Le meilleur résultat web indique qu'il pourrait correspondre à: {title}. {selected_snippet}"
        )
    else:
        description = (
            f"Produit détecté à partir du code-barres {barcode}. "
            f"Les résultats web indiquent qu'il pourrait correspondre à: {title}. "
            "Description à vérifier avant validation."
        )

    if not certain:
        description = f"{description} {UNCERTAIN_WARNING}"

    return description[:900].strip()


def search_product_info_with_serpapi(barcode: str) -> dict[str, Any]:
    barcode = str(barcode or "").strip()
    result = _empty_result(barcode)
    if not barcode:
        result["warning"] = "Code-barres vide."
        return result

    queries = [
        f"{barcode} product",
        f"{barcode} food",
        f"{barcode} produit",
        f"{barcode} marketplace",
    ]

    links: list[dict[str, str]] = []
    warnings: list[str] = []
    seen_links: set[str] = set()

    for query in queries:
        try:
            payload = _request_serpapi(
                {
                    "engine": "google",
                    "q": query,
                    "num": 10,
                    "safe": "active",
                }
            )
        except Exception as exc:
            warnings.append(str(exc))
            continue

        for item in payload.get("organic_results") or []:
            link = _clean_text(item.get("link"))
            title = _clean_title(item.get("title", ""), barcode)
            snippet = _clean_text(item.get("snippet"))
            if not link or link in seen_links or not title:
                continue
            seen_links.add(link)
            links.append({"title": title, "link": link, "snippet": snippet})

    if not links:
        result["warning"] = warnings[0] if warnings else "Aucun résultat web trouvé via SerpAPI."
        return result

    selected_result = _select_best_result(links, barcode)
    title = selected_result.get("title") or f"Produit détecté - {barcode}"
    selected_score = _result_score(selected_result, barcode, 0)[0]
    certain = selected_score >= 250 and not _is_generic_title(title)
    result.update(
        {
            "success": True,
            "product_title": title,
            "product_description": _build_description(barcode, title, selected_result, certain),
            "links": links[:8],
            "warning": "" if certain else UNCERTAIN_WARNING,
        }
    )
    return result


def search_product_images_with_serpapi(query: str, max_images: int = 6) -> list[dict[str, str]]:
    query = _clean_text(query)
    if not query:
        return []

    payload = _request_serpapi(
        {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "ijn": "0",
            "safe": "active",
        }
    )

    images: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in payload.get("images_results") or []:
        url = _clean_text(item.get("original") or item.get("thumbnail"))
        if not url or url in seen_urls or not url.startswith(("http://", "https://")):
            continue
        seen_urls.add(url)
        images.append(
            {
                "url": url,
                "title": _clean_text(item.get("title")) or query,
                "source": "SerpAPI Google Images",
            }
        )
        if len(images) >= max_images:
            break
    return images


def enrich_product_from_barcode(barcode: str) -> dict[str, Any]:
    barcode = str(barcode or "").strip()
    product_info = search_product_info_with_serpapi(barcode)
    if not product_info.get("success"):
        return product_info

    image_query = product_info.get("product_title") or f"{barcode} product"
    try:
        images = search_product_images_with_serpapi(image_query, max_images=6)
    except Exception as exc:
        images = []
        product_info["warning"] = str(exc)

    product_info["images"] = images
    if not images:
        existing_warning = product_info.get("warning", "")
        fallback_warning = "Aucune image web trouvée. Utilisation de l’image uploadée comme fallback."
        product_info["warning"] = f"{existing_warning} {fallback_warning}".strip()

    return product_info
