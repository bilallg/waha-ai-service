# barcode_to_data.py

def barcode_to_data(barcode):
    """
    Transforme le code-barres en informations exploitables pour WAHA IA.
    Renvoie un dictionnaire contenant le nom du produit, l'ID et la catégorie.
    """
    # Table de correspondance : code-barres -> infos produit
    mapping = {
        "5449000006288": {
            "product_name": "Chocolat Lait Milka 100g",
            "product_id": "MILKA001",
            "category": "Chocolat",
            "url": None
        },
        "6111248360130": {
            "product_name": "Savon Doux Dove 200ml",
            "product_id": "DOVE001",
            "category": "Hygiène",
            "url": None
        },
        "1234567890123": {
            "product_name": "Biscuit Petit Beurre 250g",
            "product_id": "PB001",
            "category": "Biscuits",
            "url": None
        }
        # Ajoute ici d'autres codes-barres réels que tu veux gérer
    }

    # Récupérer les infos du produit
    data = mapping.get(barcode)
    if not data:
        print(f"Produit inconnu pour le code-barres : {barcode}")
    return data