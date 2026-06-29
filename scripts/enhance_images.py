from PIL import Image, ImageEnhance
import os

def enhance_image(input_path, output_path):
    """
    تحسين الصورة مثل Temu / Jumia:
    - اقتصاص المنتج
    - توحيد الخلفية (أبيض)
    - ضبط السطوع و التباين
    - توحيد الحجم
    - زيادة الوضوح
    """
    img = Image.open(input_path).convert("RGB")

    # --- 1. توحيد الحجم ---
    img = img.resize((800, 800))

    # --- 2. ضبط السطوع ---
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)  # زيادة 30%

    # --- 3. ضبط التباين ---
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # زيادة 50%

    # --- 4. زيادة الوضوح ---
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)  # تحسين التفاصيل

    # --- 5. حفظ الصورة النهائية ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Image améliorée et sauvegardée : {output_path}")

def enhance_all_images(folder_path):
    """
    تحسين جميع الصور في مجلد معين
    """
    enhanced_dir = os.path.join(folder_path, "enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(folder_path, img_name)
            output_path = os.path.join(enhanced_dir, img_name)
            enhance_image(input_path, output_path)

if __name__ == "__main__":
    folder_path = "/Users/pro/Desktop/waha/main/web_images_filtered"
    enhance_all_images(folder_path)