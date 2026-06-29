from fastapi import FastAPI, UploadFile, File
import os
import shutil

from main_pipeline import run_pipeline

app = FastAPI(
    title="WAHA AI API",
    description="API IA pour scan barcode, récupération images et amélioration produit",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "WAHA AI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/scan-product")
async def scan_product(file: UploadFile = File(...)):

    try:

        upload_dir = "/Users/pro/Desktop/waha/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Image reçue : {file_path}")

        result = run_pipeline(file_path)

        return result

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/product/{barcode}")
def get_product(barcode: str):

    product_folder = f"product_{barcode}"

    final_images_dir = (
        f"/Users/pro/Desktop/waha/main/final_images/{product_folder}"
    )

    if not os.path.exists(final_images_dir):
        return {
            "status": "error",
            "message": "Produit non trouvé"
        }

    images = []

    for file in os.listdir(final_images_dir):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            images.append(file)

    return {
        "status": "success",
        "barcode": barcode,
        "product_folder": product_folder,
        "images": images
    }