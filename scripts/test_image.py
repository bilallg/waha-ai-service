import os

def check_image(image_path):
    if os.path.exists(image_path):
        print(f"Le fichier {image_path} existe !")
    else:
        print(f"Erreur : Le fichier {image_path} n'existe pas.")

if __name__ == "__main__":
    check_image("images/img.png")  # Utilise le même chemin que dans ton projet
    