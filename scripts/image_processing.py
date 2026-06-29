# image_processing.py
from PIL import Image, ImageEnhance
import os

def process_image(input_path, output_path):
    img = Image.open(input_path)
    img = img.resize((800, 800))  # Redimensionner
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print("Image traitée et sauvegardée :", output_path)
    return output_path