from __future__ import annotations

import os
import secrets
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UPLOADS_DIR = PROJECT_ROOT / "output" / "uploads"
MODEL_PATH = PROJECT_ROOT / "models" / "barcode_yolov8.pt"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

load_dotenv(PROJECT_ROOT / ".env")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
WAHA_AI_API_KEY = os.getenv("WAHA_AI_API_KEY", "").strip()

from main_pipeline import process_barcode_image


app = FastAPI(
    title="WAHA AI Service",
    description="YOLOv8 barcode detection, ZBar decoding, and SerpAPI product enrichment for WAHA.",
    version="1.0.0",
)


@app.middleware("http")
async def reject_unauthorized_ai_requests(request: Request, call_next):
    if request.url.path == "/ai/barcode/detect" and request.method.upper() == "POST":
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


def _shape_response(result: dict) -> dict:
    success = result.get("status") == "success" and bool(result.get("barcode"))
    warning = result.get("warning") or "; ".join(result.get("warnings") or [])
    message = result.get("message") or ("Barcode detected and enriched." if success else "Barcode detection failed.")
    if warning and success:
        message = f"{message} {warning}"

    return {
        "success": success,
        "barcode": result.get("barcode", ""),
        "barcode_type": result.get("barcode_type", ""),
        "product_title": result.get("product_title", ""),
        "product_description": result.get("product_description", ""),
        "images": result.get("images") or [],
        "links": result.get("links") or [],
        "source": "YOLOv8 + ZBar + SerpAPI",
        "message": message,
    }


@app.get("/health")
def health() -> dict:
    return {
        "success": True,
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
        "serpapi_configured": bool(SERPAPI_API_KEY),
        "auth_configured": bool(WAHA_AI_API_KEY),
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

        result = process_barcode_image(upload_path)
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
                "images": [],
                "links": [],
                "source": "YOLOv8 + ZBar + SerpAPI",
                "message": f"AI service error: {exc}",
            },
        )
    finally:
        await file.close()
        upload_path.unlink(missing_ok=True)
