from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

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


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

from chatbot.chatbot_recommendation import answer_seller_question
from evaluate_decoding import compare_decoding_methods, evaluate_decoding_folder
from evaluate_yolo_barcode import evaluate_yolo_model
from export_marketplace import export_marketplace_package
from main_pipeline import (
    PRODUCTS_HISTORY_PATH,
    UPLOADS_DIR,
    clean_all_results,
    load_products_history,
    process_barcode_image,
    run_pipeline,
)


st.set_page_config(page_title="WAHA Platform", page_icon="🛍️", layout="wide")

CUSTOM_CSS = """
<style>
  .main .block-container {max-width: 1400px; padding-top: 1.2rem;}
  .waha-header {padding: 1.2rem 1.4rem; border: 1px solid #e5e7eb; border-radius: 14px;
                background: linear-gradient(135deg, #fff7ed 0%, #ffffff 55%, #eff6ff 100%); margin-bottom: 1rem;}
  .waha-header h1 {margin: 0; font-size: 2rem; color: #111827;}
  .waha-header p {margin: .35rem 0 0; color: #4b5563;}
  .method-badge {display: inline-block; background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa;
                 border-radius: 999px; padding: .3rem .7rem; font-weight: 700;}
  .ok {color: #047857; font-weight: 700;}
  .error {color: #b91c1c; font-weight: 700;}
</style>
"""

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]


def save_uploaded_file(uploaded_file: Any, prefix: str = "") -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = Path(getattr(uploaded_file, "name", "barcode_upload.jpg")).name
    suffix = Path(raw_name).suffix or ".jpg"
    stem = Path(raw_name).stem or "barcode_upload"
    output_path = UPLOADS_DIR / f"{prefix}{stem}_{uuid4().hex[:8]}{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def existing_paths(paths: list[str] | None) -> list[Path]:
    return [Path(path) for path in paths or [] if path and Path(path).exists()]


def image_gallery(title: str, paths: list[str] | None, empty_message: str, columns_count: int = 4) -> None:
    st.subheader(title)
    images = existing_paths(paths)
    if not images:
        st.info(empty_message)
        return
    columns = st.columns(columns_count)
    for index, path in enumerate(images):
        with columns[index % columns_count]:
            st.image(str(path), caption=path.name, use_container_width=True)


def show_warnings(result: dict[str, Any]) -> None:
    for warning in dict.fromkeys(result.get("warnings") or []):
        st.warning(warning)


def render_serpapi_enrichment(result: dict[str, Any]) -> None:
    serpapi_result = result.get("serpapi_result") or {}
    barcode = result.get("barcode") or serpapi_result.get("barcode") or ""
    if not barcode:
        return

    st.subheader("Enrichissement produit web")
    st.write(f"**Barcode détecté:** {barcode}")
    st.write("**Mode utilisé:** YOLOv8 + pyzbar + SerpAPI")

    if not SERPAPI_API_KEY:
        st.warning(
            "SERPAPI_API_KEY manquante. Le barcode est détecté, mais les informations produit/images web ne peuvent pas être récupérées."
        )

    product_title = serpapi_result.get("product_title") or f"Produit détecté - {barcode}"
    product_description = serpapi_result.get("product_description") or "Description produit web non disponible."
    st.text_input("Product title", product_title, disabled=True)
    st.text_area("Product description", product_description, height=130, disabled=True)
    st.caption(f"Source utilisée: {serpapi_result.get('source') or 'SerpAPI'}")

    if serpapi_result.get("warning"):
        st.warning(serpapi_result["warning"])

    images = (serpapi_result.get("images") or [])[:6]
    st.subheader("Images SerpAPI")
    if images:
        columns = st.columns(3)
        for index, image in enumerate(images):
            with columns[index % 3]:
                caption = image.get("title") or image.get("source") or "SerpAPI Google Images"
                st.image(image.get("url"), caption=caption, use_container_width=True)
    else:
        st.info("Aucune image web trouvée. Utilisation de l’image uploadée comme fallback.")

    links = serpapi_result.get("links") or []
    st.subheader("Liens trouvés via SerpAPI")
    if links:
        st.dataframe(pd.DataFrame(links), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun lien SerpAPI disponible.")

    with st.expander("Résultat JSON complet"):
        st.json(result)


def barcode_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    success = result.get("status") == "success" and bool(result.get("barcode"))
    return {
        "success": success,
        "barcode": result.get("barcode", ""),
        "barcode_type": result.get("barcode_type", ""),
        "product_title": result.get("product_title", ""),
        "product_description": result.get("product_description", ""),
        "original_description": result.get("original_description", ""),
        "product_category": None,
        "price": None,
        "expiration_date": result.get("expiration_date"),
        "expiration_text": result.get("expiration_text", ""),
        "expiration_confidence": result.get("expiration_confidence"),
        "expiration_found": bool(result.get("expiration_found")),
        "images": result.get("images") or [],
        "links": result.get("links") or [],
        "source": result.get("source") or "YOLOv8 + ZBar + SerpAPI",
        "message": result.get("message") or ("Barcode detected and enriched." if success else "Barcode detection failed."),
    }


def render_barcode_result(result: dict[str, Any] | None) -> None:
    if not result:
        return

    payload = barcode_result_payload(result)
    st.subheader("Final Result")
    if payload["success"]:
        st.success(payload["message"])
    else:
        st.error(payload["message"])

    columns = st.columns(3)
    columns[0].metric("Barcode", payload["barcode"] or "-")
    columns[1].metric("Barcode Type", payload["barcode_type"] or "-")
    columns[2].metric("Source", payload["source"] or "-")

    manual_columns = st.columns(2)
    manual_columns[0].info("Price to be completed manually")
    manual_columns[1].info("Category to be selected by the seller")

    st.subheader("Expiration Date")
    if payload["expiration_found"]:
        st.success(f"Expiration date detected: {payload['expiration_date']}")
        if payload["expiration_confidence"] is not None:
            st.caption(f"OCR confidence: {float(payload['expiration_confidence']):.0%}")
        if payload["expiration_text"]:
            st.text_input("OCR text match", payload["expiration_text"], disabled=True, key="scan_expiration_text")
    else:
        st.warning("Expiration date not detected. Please take a clearer image of the printed date.")
        if payload["expiration_text"]:
            with st.expander("OCR text read from image"):
                st.write(payload["expiration_text"])

    st.text_input("Product title", payload["product_title"], disabled=True, key="scan_product_title")
    st.text_area("Product description", payload["product_description"], height=120, disabled=True, key="scan_product_description")
    if payload["original_description"]:
        with st.expander("Original description"):
            st.write(payload["original_description"])

    images = payload["images"][:6]
    if images:
        st.subheader("Images")
        image_columns = st.columns(3)
        for index, image in enumerate(images):
            with image_columns[index % 3]:
                st.image(
                    image.get("url"),
                    caption=image.get("title") or image.get("source") or "Product image",
                    use_container_width=True,
                )

    if payload["links"]:
        st.subheader("Links")
        st.dataframe(pd.DataFrame(payload["links"]), use_container_width=True, hide_index=True)

    with st.expander("API-ready JSON"):
        st.json(payload)


def process_barcode_input(image_path: str | Path) -> None:
    with st.spinner("Processing barcode..."):
        try:
            st.session_state.barcode_scan_result = process_barcode_image(image_path)
        except Exception as exc:
            st.session_state.barcode_scan_result = {
                "status": "error",
                "success": False,
                "message": f"Unexpected barcode processing error: {exc}",
                "barcode": "",
                "barcode_type": "",
                "product_title": "",
                "product_description": "",
                "original_description": "",
                "product_category": None,
                "price": None,
                "expiration_date": None,
                "expiration_text": "",
                "expiration_confidence": None,
                "expiration_found": False,
                "images": [],
                "links": [],
                "source": "YOLOv8 + ZBar + SerpAPI",
            }


def render_barcode_input_modes() -> None:
    upload_tab, camera_tab = st.tabs(["Upload Image", "Scan with Camera"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Image contenant le code-barres",
            type=IMAGE_TYPES,
            key="pipeline_upload",
        )
        if uploaded:
            try:
                saved = save_uploaded_file(uploaded, "upload_")
                st.session_state.uploaded_image_path = str(saved)
                st.image(str(saved), caption="Image chargée", use_container_width=True)
                if st.button("Process uploaded image", type="primary", use_container_width=True, key="process_upload_barcode"):
                    process_barcode_input(saved)
            except Exception as exc:
                st.error(f"Impossible d'enregistrer l'image: {exc}")
        else:
            st.info("Upload an image containing a product barcode.")

    with camera_tab:
        st.write("Place the barcode in front of the camera and take a photo.")
        captured = st.camera_input("Scan with Camera", key="camera_barcode_input")
        if captured:
            try:
                saved = save_uploaded_file(captured, "camera_")
                st.session_state.uploaded_image_path = str(saved)
                st.image(str(saved), caption="Photo capturée", use_container_width=True)
                if st.button("Process camera photo", type="primary", use_container_width=True, key="process_camera_barcode"):
                    process_barcode_input(saved)
            except Exception as exc:
                st.error(f"Impossible d'enregistrer la photo: {exc}")

    render_barcode_result(st.session_state.get("barcode_scan_result"))


def approval_key(result: dict[str, Any]) -> str:
    return f"seller_approval_{result.get('barcode', 'unknown')}"


def approval_payload(result: dict[str, Any]) -> dict[str, Any]:
    decisions = st.session_state.get(approval_key(result), {})
    return {
        "barcode": result.get("barcode", ""),
        "product_folder": result.get("product_folder", ""),
        "accepted_images": [path for path, accepted in decisions.items() if accepted],
        "rejected_images": [path for path, accepted in decisions.items() if not accepted],
    }


def save_approval(result: dict[str, Any]) -> Path:
    export_dir = Path(result.get("export_dir") or PROJECT_ROOT / "output" / "exports" / result["product_folder"])
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / "seller_approval.json"
    output_path.write_text(json.dumps(approval_payload(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def render_approval(result: dict[str, Any]) -> None:
    images = existing_paths(result.get("final_images"))
    if not images:
        st.info("Aucune image finale disponible.")
        return
    key = approval_key(result)
    if key not in st.session_state:
        st.session_state[key] = {str(path): True for path in images}

    columns = st.columns(4)
    for index, path in enumerate(images):
        with columns[index % 4]:
            with st.container(border=True):
                st.image(str(path), use_container_width=True)
                accepted = st.checkbox(
                    "Approuver",
                    value=st.session_state[key].get(str(path), True),
                    key=f"{key}_{path.name}",
                )
                st.session_state[key][str(path)] = accepted
                st.caption("Acceptée" if accepted else "Refusée")

    accepted_count = sum(st.session_state[key].values())
    st.info(f"{accepted_count}/{len(images)} image(s) approuvée(s).")
    if st.button("Sauvegarder la validation vendeur", key="save_approval"):
        try:
            st.success(f"Validation sauvegardée: {save_approval(result)}")
        except Exception as exc:
            st.error(f"Impossible de sauvegarder la validation: {exc}")


def file_download(label: str, path_value: str | None, mime: str) -> None:
    path = Path(path_value) if path_value else None
    if path and path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True)
    else:
        st.button(label, disabled=True, use_container_width=True)


def render_dashboard(result: dict[str, Any] | None) -> None:
    st.subheader("Entrée vendeur")
    render_barcode_input_modes()

    st.divider()
    st.subheader("Pipeline marketplace complet")
    left, right = st.columns([1, 1])
    with left:
        use_yolo = st.checkbox(
            "Utiliser YOLOv8 pour détecter le code-barres",
            value=True,
            key="use_yolo",
        )
        max_images = st.slider("Nombre maximal d'images web", 3, 30, 30)
        run_clicked = st.button("Lancer pipeline WAHA", type="primary", use_container_width=True)
    with right:
        preview = st.session_state.get("uploaded_image_path")
        if preview and Path(preview).exists():
            st.image(preview, caption="Image chargée", use_container_width=True)
        else:
            st.info("Charge une image pour démarrer.")

    if run_clicked:
        image_path = st.session_state.get("uploaded_image_path")
        if not image_path:
            st.warning("Ajoute d'abord une image contenant un code-barres.")
        else:
            status = st.status("Pipeline WAHA en cours...", expanded=True)

            def progress(message: str) -> None:
                status.write(message)

            try:
                pipeline_result = run_pipeline(
                    image_path,
                    max_images=max_images,
                    min_final_images=3,
                    clean=True,
                    use_yolo=use_yolo,
                    progress_callback=progress,
                )
            except Exception as exc:
                pipeline_result = {"status": "error", "message": f"Erreur inattendue du pipeline: {exc}", "warnings": []}
            st.session_state.pipeline_result = pipeline_result
            if pipeline_result.get("status") == "success":
                status.update(label="Pipeline terminé", state="complete", expanded=False)
            else:
                status.update(label="Pipeline interrompu proprement", state="error", expanded=True)
            st.rerun()

    if not result:
        render_history()
        return
    render_serpapi_enrichment(result)
    if result.get("status") != "success":
        st.error(result.get("message", "Le pipeline n'a pas abouti."))
        show_warnings(result)
        return

    st.markdown("<p class='ok'>Pipeline terminé avec succès</p>", unsafe_allow_html=True)
    metrics = st.columns(5)
    metrics[0].metric("Code-barres", result.get("barcode", ""))
    metrics[1].metric("Méthode", result.get("barcode_decoding_method", ""))
    metrics[2].metric("Téléchargées", result.get("downloaded_images_count", 0))
    metrics[3].metric("Filtrées", result.get("selected_images_count", 0))
    metrics[4].metric("Finales", result.get("final_images_count", 0))

    metadata = result.get("metadata") or {}
    st.subheader("Produit identifié")
    st.json(
        {
            "title": metadata.get("title", ""),
            "brand": metadata.get("brand", ""),
            "product_category": None,
            "price": None,
            "quantity": metadata.get("quantity", ""),
            "source": metadata.get("source", ""),
        },
        expanded=True,
    )
    show_warnings(result)
    render_history()


def render_yolo_detection(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("Lance le pipeline pour afficher la détection.")
        return
    columns = st.columns(4)
    columns[0].metric("Confiance YOLO", f"{float(result.get('yolo_confidence', 0)):.1%}")
    columns[1].metric("Inférence", f"{float(result.get('yolo_inference_time_ms', 0)):.2f} ms")
    columns[2].metric("Boîtes", len(result.get("yolo_boxes") or []))
    columns[3].metric("Décodage", "Réussi" if result.get("barcode_decode_success") else "Échec")

    method = result.get("barcode_decoding_method", "")
    display_method = "YOLO + ZBar" if method == "yolo_crop_zbar" else "Fallback image complète ZBar"
    st.markdown(f"<span class='method-badge'>{display_method}</span>", unsafe_allow_html=True)

    annotated = result.get("yolo_annotated_image")
    if annotated and Path(annotated).exists():
        st.subheader("Image annotée")
        st.image(annotated, use_container_width=True)
    elif result.get("barcode_detection_method") == "legacy_full_image":
        st.info("YOLO n'a pas produit d'annotation; le scanner historique a décodé l'image complète.")
    else:
        st.warning("Aucune image annotée. Le modèle est peut-être absent ou aucune boîte n'a été détectée.")

    image_gallery(
        "Crops code-barres",
        result.get("yolo_cropped_images"),
        "Aucun crop YOLO disponible.",
        columns_count=3,
    )
    if result.get("yolo_boxes"):
        st.dataframe(result["yolo_boxes"], use_container_width=True, hide_index=True)
    if result.get("barcode_decode_error") and not result.get("barcode_decode_success"):
        st.warning(result["barcode_decode_error"])


def render_marketplace(result: dict[str, Any] | None) -> None:
    if not result or result.get("status") != "success":
        st.info("Lance un pipeline réussi pour afficher la fiche marketplace.")
        return
    metadata = result.get("metadata") or {}
    product_data = result.get("product_data") or {}
    st.subheader("Résultat OpenFoodFacts")
    st.json(metadata, expanded=False)

    downloaded, filtered, final = st.tabs(["Images web", "Images filtrées", "Images finales 1000×1000"])
    with downloaded:
        image_gallery("Images téléchargées", result.get("downloaded_images"), "Aucune image web disponible.")
    with filtered:
        image_gallery("Images retenues", result.get("selected_images"), "Aucune image filtrée disponible.")
    with final:
        render_approval(result)

    st.subheader("Fiche produit générée")
    first, second = st.columns(2)
    with first:
        st.text_input("Titre SEO", product_data.get("seo_title", ""), disabled=True)
        st.text_input("Prix", "Price to be completed manually", disabled=True)
        st.text_input("Catégorie", "Category to be selected by the seller", disabled=True)
        st.text_area("Description courte", product_data.get("short_description", ""), disabled=True)
    with second:
        st.text_area("Description longue", product_data.get("long_description", ""), height=150, disabled=True)
        st.text_area("Tags", ", ".join(product_data.get("tags") or []), disabled=True)
        st.text_area("Mots-clés", ", ".join(product_data.get("seo_keywords") or []), disabled=True)


def regenerate_export(result: dict[str, Any]) -> None:
    approval = approval_payload(result)
    accepted = approval["accepted_images"] or result.get("final_images", [])
    export_paths = export_marketplace_package(
        product_data=result.get("product_data") or result.get("metadata") or {},
        final_images=accepted,
        video_path=result.get("video_path"),
        export_dir=result.get("export_dir") or PROJECT_ROOT / "output" / "exports" / result["product_folder"],
        approval=approval,
    )
    result["export_paths"] = export_paths
    result["export_dir"] = export_paths["export_dir"]
    result["export_zip"] = export_paths["zip_path"]
    st.session_state.pipeline_result = result


def render_video_export(result: dict[str, Any] | None) -> None:
    if not result or result.get("status") != "success":
        st.info("Lance un pipeline réussi pour générer les exports.")
        return
    video = result.get("video_path")
    if video and Path(video).exists():
        st.subheader("Vidéo produit")
        st.video(video)
    else:
        st.warning("La vidéo MP4 n'a pas pu être générée.")

    if st.button("Générer / exporter ZIP", type="primary", use_container_width=True):
        try:
            regenerate_export(result)
            st.success("Package marketplace régénéré.")
        except Exception as exc:
            st.error(f"Impossible de générer le package: {exc}")

    paths = result.get("export_paths") or {}
    columns = st.columns(4)
    with columns[0]:
        file_download("Télécharger product.json", paths.get("product_json"), "application/json")
    with columns[1]:
        file_download("Télécharger product.csv", paths.get("product_csv"), "text/csv")
    with columns[2]:
        file_download("Télécharger seller_approval.json", paths.get("seller_approval_json"), "application/json")
    with columns[3]:
        file_download("Télécharger ZIP marketplace", paths.get("zip_path"), "application/zip")


def evaluation_test_folder() -> Path:
    folder_text = st.text_input(
        "Dossier d'images de test",
        value=str(PROJECT_ROOT / "data" / "barcode_dataset" / "images" / "test"),
    )
    uploaded_tests = st.file_uploader(
        "Ou téléverser plusieurs images de test",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="evaluation_uploads",
    )
    if uploaded_tests:
        upload_dir = PROJECT_ROOT / "output" / "evaluation" / "uploaded_tests"
        upload_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_tests:
            (upload_dir / Path(uploaded.name).name).write_bytes(uploaded.getbuffer())
        return upload_dir
    return Path(folder_text).expanduser()


def render_evaluation() -> None:
    test_folder = evaluation_test_folder()
    st.caption(f"Jeu de test actif: {test_folder}")
    buttons = st.columns(3)
    if buttons[0].button("Évaluer YOLO model", use_container_width=True):
        with st.spinner("Évaluation YOLO en cours..."):
            try:
                st.session_state.yolo_metrics = evaluate_yolo_model()
            except Exception as exc:
                st.session_state.yolo_metrics = {"status": "error", "message": str(exc)}
    if buttons[1].button("Évaluer decoding", use_container_width=True):
        with st.spinner("Évaluation du décodage en cours..."):
            try:
                st.session_state.decoding_metrics = evaluate_decoding_folder(test_folder)
            except Exception as exc:
                st.session_state.decoding_metrics = {"status": "error", "message": str(exc)}
    if buttons[2].button("Comparer ancien vs nouveau", use_container_width=True):
        with st.spinner("Comparaison en cours..."):
            try:
                st.session_state.decoding_comparison = compare_decoding_methods(test_folder)
            except Exception as exc:
                st.session_state.decoding_comparison = {"status": "error", "message": str(exc)}

    yolo = st.session_state.get("yolo_metrics") or {}
    decoding = st.session_state.get("decoding_metrics") or {}
    if yolo.get("message") and yolo.get("status") == "error":
        st.warning(yolo["message"])
    if decoding.get("message") and decoding.get("status") == "error":
        st.warning(decoding["message"])

    metric_columns = st.columns(7)
    values = [
        ("Precision", yolo.get("precision", 0)),
        ("Recall", yolo.get("recall", 0)),
        ("F1-score", yolo.get("f1_score", 0)),
        ("mAP@0.5", yolo.get("map50", 0)),
        ("mAP@0.5:0.95", yolo.get("map50_95", 0)),
        ("Detection rate", decoding.get("detection_rate", 0)),
        ("Decoding rate", decoding.get("decoding_success_rate", 0)),
    ]
    for column, (label, value) in zip(metric_columns, values):
        column.metric(label, f"{float(value):.2%}")
    st.metric(
        "Temps moyen d'inférence YOLO",
        f"{float(yolo.get('avg_inference_time_ms', 0)):.2f} ms",
    )

    charts = yolo.get("charts") or {}
    chart_columns = st.columns(3)
    for column, key, label in zip(
        chart_columns,
        ["pr_curve", "f1_curve", "confusion_matrix"],
        ["Courbe PR", "Courbe F1", "Matrice de confusion"],
    ):
        with column:
            path = charts.get(key)
            st.subheader(label)
            if path and Path(path).exists():
                st.image(path, use_container_width=True)
            else:
                st.info("Graphique non disponible.")

    comparison = st.session_state.get("decoding_comparison") or {}
    if comparison.get("status") == "success":
        old = comparison["old_method"]
        new = comparison["new_method"]
        table = pd.DataFrame(
            [
                {
                    "Méthode": old["name"],
                    "Taux de succès": old["success_rate"],
                    "Temps moyen (ms)": old["average_time_ms"],
                    "Échecs": len(old["failed_images"]),
                },
                {
                    "Méthode": new["name"],
                    "Taux de succès": new["success_rate"],
                    "Temps moyen (ms)": new["average_time_ms"],
                    "Échecs": len(new["failed_images"]),
                },
            ]
        )
        st.subheader("Comparaison pyzbar seul vs YOLO + ZBar")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("Méthode")[["Taux de succès"]])

    results_csv = PROJECT_ROOT / "output" / "evaluation" / "decoding_results.csv"
    if results_csv.exists() and results_csv.stat().st_size:
        st.subheader("Table des résultats")
        try:
            st.dataframe(pd.read_csv(results_csv), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Impossible de lire le tableau d'évaluation: {exc}")


def render_chatbot(result: dict[str, Any] | None) -> None:
    st.caption("Le chatbot utilise la fiche product_data.json courante, ou la plus récente si aucun pipeline n'est chargé.")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Je réponds sur la catégorie, le titre, le prix, les tags, les images et les produits similaires.",
            }
        ]
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt = st.chat_input("Exemple: quelles images dois-je garder ?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        try:
            response = answer_seller_question(prompt, result)
        except Exception as exc:
            response = f"Le chatbot ne peut pas répondre pour le moment: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()


def render_history() -> None:
    st.subheader("Historique récent")
    history = load_products_history()
    if not history:
        st.info(f"Aucun historique. Fichier cible: {PRODUCTS_HISTORY_PATH}")
        return
    st.dataframe(history[:20], use_container_width=True, hide_index=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="waha-header">
          <h1>WAHA Platform</h1>
          <p>Localisation YOLOv8, décodage ZBar, génération produit et évaluation du pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Contrôles")
        if st.button("Nettoyer les résultats générés", use_container_width=True):
            try:
                clean_all_results()
                for key in ["pipeline_result", "uploaded_image_path", "yolo_metrics", "decoding_metrics", "decoding_comparison"]:
                    st.session_state.pop(key, None)
                st.success("Résultats nettoyés.")
            except Exception as exc:
                st.error(f"Nettoyage impossible: {exc}")
        st.caption("YOLO est optionnel. Sans modèle, WAHA revient automatiquement au scan ZBar de l'image complète.")

    result = st.session_state.get("pipeline_result")
    tabs = st.tabs(
        [
            "Dashboard",
            "YOLO Detection",
            "Product Marketplace",
            "Video & Export",
            "Evaluation",
            "Chatbot vendeur",
        ]
    )
    with tabs[0]:
        render_dashboard(result)
    with tabs[1]:
        render_yolo_detection(result)
    with tabs[2]:
        render_marketplace(result)
    with tabs[3]:
        render_video_export(result)
    with tabs[4]:
        render_evaluation()
    with tabs[5]:
        render_chatbot(result)


if __name__ == "__main__":
    main()
