from __future__ import annotations

import os
import secrets
import shutil
import sys
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UPLOADS_DIR = PROJECT_ROOT / "output" / "uploads"
MODEL_PATH = PROJECT_ROOT / "models" / "barcode_yolov8.pt"
SERVICE_VERSION = "1.0.1"
BUILD_TIMESTAMP = os.getenv("RENDER_GIT_COMMIT", "").strip() or datetime.now(timezone.utc).isoformat(timespec="seconds")
MAX_UPLOAD_IMAGE_SIDE = 1600

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

load_dotenv(PROJECT_ROOT / ".env")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
WAHA_AI_API_KEY = os.getenv("WAHA_AI_API_KEY", "").strip()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from expiration_date_ocr import detect_expiration_date
from main_pipeline import process_barcode_image
from openai_product_content import clean_product_title
from serpapi_product_search import (
    build_serpapi_image_query,
    search_images_for_product_title,
    search_product_links_with_serpapi,
    serpapi_configured,
)
from stock_prediction import predict_stock_depletion


app = FastAPI(
    title="WAHA AI Service",
    description="YOLOv8 barcode detection, ZBar decoding, and SerpAPI product enrichment for WAHA.",
    version="1.0.0",
)


@app.middleware("http")
async def reject_unauthorized_ai_requests(request: Request, call_next):
    protected_paths = {"/ai/barcode/detect", "/ai/expiration/detect", "/ai/stock/predict"}
    if request.url.path in protected_paths and request.method.upper() == "POST":
        if request.url.path == "/ai/stock/predict":
            LOGGER.info("STOCK PREDICTION REQUEST RECEIVED")
        if not _is_authorized(request.headers.get("X-API-Key")):
            if request.url.path == "/ai/stock/predict":
                LOGGER.info("STOCK PREDICTION ERROR: invalid or missing API key")
            return _unauthorized()
    return await call_next(request)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"success": False, "message": "Invalid or missing API key"},
    )


def _is_authorized(api_key: str | None) -> bool:
    if not api_key or not WAHA_AI_API_KEY:
        return False
    return secrets.compare_digest(api_key, WAHA_AI_API_KEY)


def _safe_upload_path(filename: str | None) -> Path:
    suffix = Path(filename or "barcode_upload.jpg").suffix.lower() or ".jpg"
    safe_suffix = suffix if len(suffix) <= 10 else ".jpg"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR / f"ai_service_{uuid4().hex}{safe_suffix}"


def _image_size(path: str | Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _short_error(exc: Exception, max_length: int = 240) -> str:
    error = f"{type(exc).__name__}: {exc}".strip()
    return error[:max_length]


def _barcode_internal_error(exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "barcode": "",
            "barcode_type": "",
            "product_title": "",
            "product_description": "",
            "expiration_date": None,
            "expiration_text": None,
            "expiration_found": False,
            "images": [],
            "links": [],
            "source": "YOLOv8 + ZBar + SerpAPI + OpenAI",
            "message": "Internal error during barcode detection",
            "error": _short_error(exc),
        },
    )


def _resize_large_image(path: str | Path, max_side: int = MAX_UPLOAD_IMAGE_SIDE) -> None:
    with Image.open(path) as image:
        image.load()
        if max(image.size) <= max_side:
            return

        image.thumbnail((max_side, max_side))
        save_format = "PNG" if image.mode in {"RGBA", "LA"} else "JPEG"
        if save_format == "JPEG":
            image = image.convert("RGB")
        image.save(path, save_format, quality=90, optimize=True)


def _valid_response_images(images: object) -> list[dict]:
    if not isinstance(images, list):
        return []
    valid_images: list[dict] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        url = str(image.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        valid_images.append(
            {
                "url": url,
                "title": str(image.get("title") or "").strip(),
                "source": image.get("source") or "SerpAPI Google Images",
            }
        )
    return valid_images[:6]


def _valid_response_links(links: object) -> list[dict]:
    if not isinstance(links, list):
        return []
    valid_links: list[dict] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        url = str(link.get("link") or link.get("url") or "").strip()
        title = str(link.get("title") or "").strip()
        if not url.startswith(("http://", "https://")) or not title:
            continue
        valid_links.append(
            {
                "title": title,
                "link": url,
                "snippet": str(link.get("snippet") or "").strip(),
            }
        )
    return valid_links[:8]


def _response_assets(result: dict) -> tuple[list[dict], list[dict]]:
    serpapi_result = result.get("serpapi_result") or {}
    images = _valid_response_images(result.get("images")) or _valid_response_images(serpapi_result.get("images"))
    links = _valid_response_links(result.get("links")) or _valid_response_links(serpapi_result.get("links"))
    return images, links


def _enrich_missing_response_assets(result: dict, product_title: str) -> tuple[list[dict], list[dict], str]:
    barcode = str(result.get("barcode") or "").strip()
    images, links = _response_assets(result)
    debug_image_query = build_serpapi_image_query(product_title, barcode)

    LOGGER.info("SERPAPI_CONFIGURED: %s", serpapi_configured())

    if not links and barcode:
        try:
            link_query = " ".join(part for part in [barcode, product_title] if part).strip()
            LOGGER.info("SERPAPI PRODUCT SEARCH START")
            links = search_product_links_with_serpapi(link_query or barcode, barcode=barcode, max_links=5)
            LOGGER.info("SERPAPI PRODUCT LINKS COUNT: %s", len(links))
        except Exception as exc:
            LOGGER.info("SERPAPI PRODUCT SEARCH FAILED: %s", exc)
            links = []

    if not images:
        try:
            images, debug_image_query = search_images_for_product_title(product_title, barcode=barcode, max_images=6)
        except Exception as exc:
            LOGGER.info("SERPAPI IMAGE SEARCH FAILED: %s", exc)
            images = []

    serpapi_result = result.get("serpapi_result")
    if isinstance(serpapi_result, dict):
        if images:
            serpapi_result["images"] = images
        if links:
            serpapi_result["links"] = links
    if images:
        result["images"] = images
    if links:
        result["links"] = links

    LOGGER.info("FINAL IMAGES COUNT: %s", len(images))
    LOGGER.info("FINAL LINKS COUNT: %s", len(links))
    return images, links, debug_image_query


def _shape_response(result: dict) -> dict:
    success = result.get("status") == "success" and bool(result.get("barcode"))
    product_title = clean_product_title(
        result.get("product_title", ""),
        {
            **(result.get("product_lookup") or {}),
            "barcode": result.get("barcode", ""),
            "quantity": (result.get("product_lookup") or {}).get("quantity", ""),
            "product_title": result.get("product_title", ""),
        },
    )
    if success:
        images, links, debug_image_query = _enrich_missing_response_assets(result, product_title)
    else:
        images, links = _response_assets(result)
        debug_image_query = ""
    message = (
        "Barcode detected and enriched."
        if success and result.get("expiration_found")
        else "Expiration date not detected"
        if success
        else result.get("message", "Barcode detection failed.")
    )

    return {
        "success": success,
        "barcode": result.get("barcode", ""),
        "barcode_type": result.get("barcode_type", ""),
        "product_title": product_title,
        "product_description": result.get("product_description", ""),
        "expiration_date": result.get("expiration_date"),
        "expiration_text": result.get("expiration_text"),
        "expiration_found": bool(result.get("expiration_found")),
        "images": images,
        "links": links,
        "source": result.get("source") or "YOLOv8 + ZBar + SerpAPI + OpenAI",
        "message": message,
        "debug_images_count": len(images),
        "debug_links_count": len(links),
        "debug_image_query": debug_image_query,
    }


@app.get("/health")
def health() -> dict:
    return {
        "success": True,
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "serpapi_configured": bool(os.getenv("SERPAPI_API_KEY", "").strip()),
        "auth_configured": bool(os.getenv("WAHA_AI_API_KEY", "").strip()),
        "stock_prediction_available": True,
        "version": SERVICE_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
    }


@app.post("/ai/stock/predict", response_model=None)
async def predict_stock(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    try:
        if not _is_authorized(x_api_key):
            return _unauthorized()

        try:
            payload = await request.json()
        except Exception as exc:
            response = {
                "success": False,
                "message": f"Invalid input: request body must be valid JSON ({_short_error(exc)})",
            }
            LOGGER.info("STOCK PREDICTION ERROR: %s", response["message"])
            return JSONResponse(status_code=422, content=response)

        response = predict_stock_depletion(payload)
        if response.get("success"):
            LOGGER.info("STOCK PREDICTION SUCCESS")
            return JSONResponse(status_code=200, content=response)

        LOGGER.info("STOCK PREDICTION ERROR: %s", response.get("message", "Invalid input"))
        return JSONResponse(status_code=422, content=response)
    except Exception as exc:
        LOGGER.exception("STOCK PREDICTION ERROR: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Invalid input: unexpected stock prediction error: {_short_error(exc)}",
            },
        )


@app.post("/ai/barcode/detect", response_model=None)
async def detect_barcode(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    upload_path: Path | None = None
    try:
        LOGGER.info("REQUEST RECEIVED endpoint=/ai/barcode/detect filename=%s", getattr(file, "filename", ""))
        if not _is_authorized(x_api_key):
            return _unauthorized()

        upload_path = _safe_upload_path(file.filename)
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        LOGGER.info("FILE READ OK path=%s size_bytes=%s", upload_path, upload_path.stat().st_size if upload_path.exists() else 0)
        _resize_large_image(upload_path)
        image_size = _image_size(upload_path)
        LOGGER.info("IMAGE DECODE OK")
        LOGGER.info("IMAGE RECEIVED endpoint=/ai/barcode/detect filename=%s path=%s size_bytes=%s", file.filename, upload_path, upload_path.stat().st_size if upload_path.exists() else 0)
        LOGGER.info("IMAGE SIZE=%s", image_size)

        LOGGER.info("BARCODE DETECTION START")
        result = process_barcode_image(upload_path)
        LOGGER.info(
            "Barcode result success=%s yolo_detections=%s pyzbar_decoded_count=%s barcode=%s",
            result.get("status") == "success",
            len(result.get("yolo_boxes") or []),
            result.get("pyzbar_decoded_count", 0),
            result.get("barcode", ""),
        )
        LOGGER.info("FULL IMAGE PYZBAR COUNT=%s", result.get("full_image_pyzbar_count", 0))
        LOGGER.info("YOLO DETECTIONS COUNT=%s", len(result.get("yolo_boxes") or []))
        LOGGER.info("CROP PYZBAR COUNT=%s", result.get("crop_pyzbar_count", 0))
        LOGGER.info("FINAL BARCODE=%s", result.get("barcode", ""))
        response = _shape_response(result)
        LOGGER.info("FINAL RESPONSE READY")
        LOGGER.info("FINAL RESPONSE READY success=%s barcode=%s", response["success"], response.get("barcode", ""))
        return JSONResponse(status_code=200 if response["success"] else 422, content=response)
    except Exception as exc:
        return _barcode_internal_error(exc)
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if upload_path:
            upload_path.unlink(missing_ok=True)


def _shape_expiration_response(result: dict) -> dict:
    found = bool(result.get("expiration_found"))
    return {
        "success": True,
        "expiration_date": result.get("expiration_date"),
        "expiration_text": result.get("expiration_text") if found else None,
        "expiration_found": found,
        "message": "Expiration date detected" if found else "Expiration date not detected",
    }


@app.post("/ai/expiration/detect", response_model=None)
async def detect_expiration(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    if not _is_authorized(x_api_key):
        return _unauthorized()

    upload_path = _safe_upload_path(file.filename)
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        LOGGER.info(
            "Image received endpoint=/ai/expiration/detect filename=%s path=%s size_bytes=%s image_size=%s",
            file.filename,
            upload_path,
            upload_path.stat().st_size if upload_path.exists() else 0,
            _image_size(upload_path),
        )

        result = detect_expiration_date(upload_path)
        return JSONResponse(status_code=200, content=_shape_expiration_response(result))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "expiration_date": None,
                "expiration_text": None,
                "expiration_found": False,
                "message": f"AI service error: {exc}",
            },
        )
    finally:
        await file.close()
        upload_path.unlink(missing_ok=True)
