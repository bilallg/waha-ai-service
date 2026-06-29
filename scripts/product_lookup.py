from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


DEFAULT_TIMEOUT = 10


class ProductProvider(Protocol):
    name: str

    def lookup(self, barcode: str) -> dict[str, str] | None:
        ...


def _clean_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _first_value(product: dict, keys: list[str], fallback: str = "") -> str:
    for key in keys:
        value = _clean_text(product.get(key))
        if value:
            return value
    return fallback


@dataclass(frozen=True)
class OpenFoodFactsProvider:
    name: str = "OpenFoodFacts"
    api_base: str = "https://world.openfoodfacts.org/api/v2/product"

    def lookup(self, barcode: str) -> dict[str, str] | None:
        url = f"{self.api_base}/{barcode}.json"
        response = requests.get(
            url,
            params={
                "fields": ",".join(
                    [
                        "product_name",
                        "product_name_fr",
                        "generic_name",
                        "generic_name_fr",
                        "brands",
                        "categories",
                        "categories_tags",
                        "quantity",
                        "image_front_url",
                        "image_url",
                    ]
                )
            },
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "WAHA-Platform/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != 1:
            return None

        product = payload.get("product") or {}
        title = _first_value(product, ["product_name_fr", "product_name", "generic_name_fr", "generic_name"])
        if not title:
            return None

        categories = _clean_text(product.get("categories"))
        if not categories:
            category_tags = product.get("categories_tags") or []
            categories = ", ".join(str(item).replace("en:", "").replace("fr:", "") for item in category_tags[:3])

        description = _first_value(product, ["generic_name_fr", "generic_name"], title)

        return {
            "barcode": barcode,
            "title": title,
            "brand": _clean_text(product.get("brands")),
            "category": categories or "Categorie provisoire",
            "description": description,
            "quantity": _clean_text(product.get("quantity")),
            "image_url": _first_value(product, ["image_front_url", "image_url"]),
            "source": self.name,
        }


@dataclass(frozen=True)
class PlaceholderProvider:
    name: str = "fallback"

    def lookup(self, barcode: str) -> dict[str, str] | None:
        return fallback_product(barcode)


def fallback_product(barcode: str) -> dict[str, str]:
    return {
        "barcode": barcode,
        "title": f"Produit {barcode}",
        "brand": "",
        "category": "Categorie provisoire",
        "description": "Description provisoire a valider par le vendeur",
        "quantity": "",
        "image_url": "",
        "source": "fallback",
    }


def lookup_product(
    barcode: str,
    providers: list[ProductProvider] | None = None,
) -> dict[str, str]:
    cleaned_barcode = str(barcode).strip()
    lookup_providers: list[ProductProvider] = providers or [OpenFoodFactsProvider()]

    for provider in lookup_providers:
        try:
            product = provider.lookup(cleaned_barcode)
        except Exception as exc:
            print(f"{provider.name} indisponible pour {cleaned_barcode}: {exc}")
            continue
        if product:
            return {**fallback_product(cleaned_barcode), **product, "source": provider.name}

    return fallback_product(cleaned_barcode)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Identifier un produit depuis OpenFoodFacts.")
    parser.add_argument("barcode", help="Code-barres a rechercher")
    args = parser.parse_args()

    print(json.dumps(lookup_product(args.barcode), indent=2, ensure_ascii=False))
