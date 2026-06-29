# WAHA Platform

WAHA transforme une image vendeur contenant un code-barres en fiche produit marketplace. YOLOv8 localise le code-barres, ZBar/pyzbar le décode, puis le pipeline identifie le produit, recherche et filtre des images, produit des visuels 1000×1000, génère une vidéo, exporte JSON/CSV/ZIP et alimente le chatbot vendeur.

YOLO reste optionnel. Si le modèle manque, si aucune boîte n'est détectée ou si le crop n'est pas décodable, WAHA tente automatiquement ZBar/pyzbar sur l'image complète puis conserve le scanner historique comme dernier fallback.

## Pipeline

```text
Image uploadée
→ détection YOLOv8 (optionnelle)
→ crop de la zone code-barres
→ décodage ZBar/pyzbar
→ fallback ZBar sur image complète
→ OpenFoodFacts
→ recherche SerpAPI
→ filtrage ResNet50 ou score qualité local
→ suppression du fond ou fallback blanc
→ images marketplace 1000×1000
→ vidéo MP4
→ description, tags, catégorie et mots-clés
→ export JSON/CSV/ZIP
→ historique et chatbot vendeur
```

YOLO sert uniquement à la localisation. La lecture de la valeur EAN/UPC/QR est effectuée par ZBar via `pyzbar`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ZBar est une dépendance système:

```bash
# macOS
brew install zbar

# Debian/Ubuntu
sudo apt-get install libzbar0
```

Pour la recherche d'images:

```bash
export SERPAPI_API_KEY="votre_cle"
```

Ou crée un fichier `.env` à la racine du projet:

```bash
SERPAPI_API_KEY=votre_cle_serpapi
WAHA_AI_API_KEY=votre_cle_interne_backend
```

`app_streamlit.py` charge automatiquement ce fichier avec `python-dotenv`. Ne commit jamais `.env` sur GitHub: il contient une clé privée et il est ignoré par `.gitignore`.

Sans clé SerpAPI, modèle YOLO, rembg ou Torch/TorchVision, le pipeline affiche un avertissement clair et utilise les fallbacks disponibles.

## Modèle YOLO

Le poids attendu est:

```text
models/barcode_yolov8.pt
```

S'il est absent, l'application continue en mode scan image complète.

## Préparer le dataset

La structure YOLO est déjà créée:

```text
data/barcode_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Chaque image doit avoir un fichier `.txt` du même nom dans le dossier `labels` correspondant. Une ligne d'annotation suit le format YOLO:

```text
class_id x_center y_center width height
```

Les coordonnées sont normalisées entre 0 et 1. Le template `data.yaml` définit une seule classe, `barcode`.

## Entraîner YOLOv8

```bash
python scripts/train_yolo_barcode.py \
  --data data/barcode_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project models/training_runs
```

Le meilleur poids est copié vers `models/barcode_yolov8.pt`. Les métriques et artefacts Ultralytics restent dans `models/training_runs/barcode_yolov8/`.

## Évaluer le modèle

```bash
python scripts/evaluate_yolo_barcode.py \
  --model models/barcode_yolov8.pt \
  --data data/barcode_dataset/data.yaml \
  --split val
```

Sorties:

```text
output/evaluation/yolo_metrics.json
output/evaluation/metrics_summary.csv
output/evaluation/charts/
```

Évaluation du décodage:

```bash
python scripts/evaluate_decoding.py data/barcode_dataset/images/test
```

Comparaison scan historique contre YOLO + ZBar:

```bash
python scripts/evaluate_decoding.py data/barcode_dataset/images/test --compare
```

Sorties:

```text
output/evaluation/decoding_metrics.json
output/evaluation/decoding_results.csv
output/evaluation/decoding_comparison.json
output/evaluation/decoding_comparison.csv
```

## Interpréter les métriques

- **Precision**: proportion des détections YOLO qui correspondent réellement à un code-barres. Une précision élevée limite les faux positifs.
- **Recall**: proportion des vrais codes-barres localisés. Un recall élevé limite les codes-barres manqués.
- **F1-score**: moyenne harmonique precision/recall. Utile pour juger leur équilibre.
- **mAP@0.5**: qualité de détection avec un seuil IoU de 0,5.
- **mAP@0.5:0.95**: moyenne sur plusieurs seuils IoU, plus stricte sur la précision des boîtes.
- **Decoding success rate**: proportion d'images pour lesquelles une valeur de code-barres est effectivement lue. Cette métrique inclut localisation, qualité du crop et capacité ZBar.
- **Average inference time**: temps moyen de détection YOLO par image.

Une excellente mAP ne garantit pas un excellent taux de décodage: une boîte peut être correcte tout en contenant un code flou, trop petit ou déformé.

## Lancer Streamlit

```bash
streamlit run app_streamlit.py
```

Test rapide avec une image:

1. Ouvre l'onglet **Dashboard**.
2. Charge une image contenant un code-barres lisible.
3. Clique sur **Lancer pipeline WAHA**.
4. Vérifie le bloc **Enrichissement produit web**.

Sortie attendue après décodage:

- Barcode détecté.
- Mode utilisé: `YOLOv8 + pyzbar + SerpAPI`.
- Titre produit estimé depuis les résultats web.
- Description courte générée à partir des snippets SerpAPI, avec avertissement si l'information est incertaine.
- Jusqu'à 6 images Google Images via SerpAPI.
- Source utilisée, liens trouvés et section **Résultat JSON complet**.

Si `SERPAPI_API_KEY` manque, si le quota SerpAPI est dépassé ou si une requête échoue, l'application ne plante pas. Le code-barres détecté reste visible et un avertissement explique que les informations produit/images web n'ont pas pu être récupérées.

## Lancer le service FastAPI

Le backend Spring Boot peut appeler le service IA séparé:

```bash
uvicorn ai_service:app --host 0.0.0.0 --port 8001 --reload
```

Endpoints:

- `GET /health`: vérification sans authentification.
- `POST /ai/barcode/detect`: analyse une image envoyée en `multipart/form-data`, champ `file`.

Le endpoint `/ai/barcode/detect` exige le header:

```text
X-API-Key: votre_cle_interne_backend
```

La valeur doit correspondre à `WAHA_AI_API_KEY` dans `.env`. Si le header manque ou ne correspond pas, le service retourne:

```json
{"success": false, "message": "Invalid or missing API key"}
```

Exemple de test:

```bash
curl -X POST "http://localhost:8001/ai/barcode/detect" \
  -H "X-API-Key: votre_cle_interne_backend" \
  -F "file=@images/eau.jpeg"
```

La réponse contient le code-barres, son type, le titre produit, la description, les images, les liens et la source `YOLOv8 + ZBar + SerpAPI`.

L'interface contient six onglets:

1. **Dashboard**: upload, option YOLO, statut du pipeline et résultat produit.
2. **YOLO Detection**: annotation, crops, confiance, temps et méthode utilisée.
3. **Product Marketplace**: OpenFoodFacts, images web/filtrées/finales et validation vendeur.
4. **Video & Export**: MP4 et téléchargements JSON/CSV/ZIP.
5. **Evaluation**: métriques YOLO, décodage, graphiques et comparaison ancien/nouveau.
6. **Chatbot vendeur**: catégorie, titre, prix, tags, images et produits similaires.

## Lancer le pipeline en terminal

```bash
python scripts/main_pipeline.py images/stylo.jpeg
python scripts/main_pipeline.py images/stylo.jpeg --no-yolo
python scripts/main_pipeline.py images/stylo.jpeg --max-images 30 --min-final 3 --no-clean
```

Les nouveaux champs de traçabilité sont inclus dans le résultat et dans `product_data.json`:

```text
barcode_detection_method
barcode_decoding_method
yolo_boxes
yolo_confidence
yolo_annotated_image
yolo_inference_time_ms
barcode_decode_success
```

## Sorties principales

```text
main/yolo_detections/product_<timestamp>/
main/web_images/product_<barcode>/
main/web_images_filtered/product_<barcode>/
main/final_images/product_<barcode>/
main/videos/product_<barcode>/product_showcase.mp4
output/exports/product_<barcode>/product_data.json
output/exports/product_<barcode>/product.json
output/exports/product_<barcode>/product.csv
output/exports/product_<barcode>/seller_approval.json
output/exports/product_<barcode>/waha_export_<barcode>.zip
data/products_history.json
```

## Gestion des erreurs

- Modèle YOLO absent: fallback scan image complète.
- ZBar absent: message d'installation, sans traceback dans l'interface.
- Aucun code détecté/décodé: arrêt propre avant la recherche produit.
- Produit OpenFoodFacts absent: fiche provisoire vendeur.
- Clé SerpAPI absente ou aucune image valide: avertissement explicite.
- rembg absent: fond blanc sans suppression IA.
- Torch/TorchVision absent: score qualité local.
- Vidéo impossible: les images et exports restent disponibles.
- Dossier d'export absent: création automatique.

Le chatbot principal utilise directement le `product_data` courant. Lancé séparément avec `streamlit run chatbot/chatbot_recommendation.py`, il charge le dernier `output/exports/product_*/product_data.json`.
