from __future__ import annotations

import os
import secrets
import shutil
import sys
import logging
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

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

load_dotenv(PROJECT_ROOT / ".env")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
WAHA_AI_API_KEY = os.getenv("WAHA_AI_API_KEY", "").strip()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from expiration_date_ocr import detect_expiration_date
from main_pipeline import process_barcode_image


app = FastAPI(
    title="WAHA AI Service",
    description="YOLOv8 barcode detection, ZBar decoding, and SerpAPI product enrichment for WAHA.",
    version="1.0.0",
)


@app.middleware("http")
async def reject_unauthorized_ai_requests(request: Request, call_next):
    if request.url.path in {"/ai/barcode/detect", "/ai/expiration/detect"} and request.method.upper() == "POST":
        if not _is_authorized(request.headers.get("X-API-Key")):
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


def _shape_response(result: dict) -> dict:
    success = result.get("status") == "success" and bool(result.get("barcode"))
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
        "product_title": result.get("product_title", ""),
        "product_description": result.get("product_description", ""),
        "expiration_date": result.get("expiration_date"),
        "expiration_text": result.get("expiration_text"),
        "expiration_found": bool(result.get("expiration_found")),
        "images": result.get("images") or [],
        "links": result.get("links") or [],
        "source": result.get("source") or "YOLOv8 + ZBar + SerpAPI",
        "message": message,
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
        "version": SERVICE_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
    }


@app.post("/ai/barcode/detect", response_model=None)
async def detect_barcode(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    if not _is_authorized(x_api_key):
        return _unauthorized()

    upload_path = _safe_upload_path(file.filename)
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_size = _image_size(upload_path)
        LOGGER.info("IMAGE RECEIVED endpoint=/ai/barcode/detect filename=%s path=%s size_bytes=%s", file.filename, upload_path, upload_path.stat().st_size if upload_path.exists() else 0)
        LOGGER.info("IMAGE SIZE=%s", image_size)

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
        return JSONResponse(status_code=200 if response["success"] else 422, content=response)
    except Exception as exc:
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
                "source": "YOLOv8 + ZBar + SerpAPI",
                "message": f"AI service error: {exc}",
            },
        )
    finally:
        await file.close()
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
