from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    from rembg import remove
except Exception as exc:
    print(f"rembg indisponible, fallback sans suppression IA: {exc}")
    remove = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def add_white_background(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background

    return image.convert("RGB")


def remove_background(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if remove is None:
        with Image.open(path) as image:
            return add_white_background(image)

    try:
        output_bytes = remove(path.read_bytes())
        with Image.open(io.BytesIO(output_bytes)) as image:
            return add_white_background(image)
    except Exception as exc:
        print(f"rembg ignore pour {path.name}: {exc}")
        with Image.open(path) as image:
            return add_white_background(image)


def auto_crop_product(image: Image.Image, padding: int = 45) -> Image.Image:
    rgb_image = image.convert("RGB")
    background = Image.new("RGB", rgb_image.size, (255, 255, 255))
    diff = ImageChops.difference(rgb_image, background).convert("L")
    bbox = diff.point(lambda pixel: 255 if pixel > 12 else 0).getbbox()

    if not bbox:
        return rgb_image

    left, top, right, bottom = bbox
    left = max(left - padding, 0)
    top = max(top - padding, 0)
    right = min(right + padding, rgb_image.width)
    bottom = min(bottom + padding, rgb_image.height)
    return rgb_image.crop((left, top, right, bottom))


def center_on_white_canvas(
    image: Image.Image,
    size: tuple[int, int] = (1000, 1000),
    product_ratio: float = 0.86,
) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    product = image.copy().convert("RGB")
    max_width = int(size[0] * product_ratio)
    max_height = int(size[1] * product_ratio)
    product.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

    x = (size[0] - product.width) // 2
    y = (size[1] - product.height) // 2
    canvas.paste(product, (x, y))
    return canvas


def enhance_marketplace_quality(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.10)
    image = ImageEnhance.Contrast(image).enhance(1.16)
    image = ImageEnhance.Color(image).enhance(1.08)
    image = ImageEnhance.Sharpness(image).enhance(1.35)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=145, threshold=3))


def enhance_image(
    input_path: str | Path,
    output_path: str | Path,
    size: tuple[int, int] = (1000, 1000),
    remove_bg: bool = True,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if remove_bg:
        image = remove_background(input_path)
    else:
        with Image.open(input_path) as source:
            image = add_white_background(source)

    image = auto_crop_product(image)
    image = center_on_white_canvas(image, size=size)
    image = enhance_marketplace_quality(image)
    image.convert("RGB").save(output, "JPEG", quality=95, optimize=True)

    print(f"Image finale sauvegardee: {output}")
    return str(output)


def improve_images_batch(
    image_paths: list[str | Path],
    final_dir: str | Path,
    min_images: int = 3,
    size: tuple[int, int] = (1000, 1000),
) -> list[str]:
    output_dir = Path(final_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_paths = [
        Path(path)
        for path in image_paths
        if Path(path).exists() and Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not valid_paths:
        return []

    total = max(min_images, len(valid_paths))
    final_images: list[str] = []

    for index in range(total):
        source = valid_paths[index % len(valid_paths)]
        output_path = output_dir / f"{index + 1}.jpg"
        final_images.append(enhance_image(source, output_path, size=size, remove_bg=True))

    return final_images


def create_product_video(
    image_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    seconds_per_image: float = 1.2,
    fps: int = 24,
    size: tuple[int, int] = (1000, 1000),
) -> str | None:
    valid_paths = [Path(path) for path in image_paths if Path(path).exists()]
    if not valid_paths:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )

    if not writer.isOpened():
        return None

    frames_per_image = max(1, int(seconds_per_image * fps))
    title_text = (title or "").strip()
    try:
        for image_path in valid_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for frame_index in range(frames_per_image):
                progress = frame_index / max(frames_per_image - 1, 1)
                zoom = 1.0 + 0.055 * progress
                zoom_width = int(size[0] / zoom)
                zoom_height = int(size[1] / zoom)
                left = max((size[0] - zoom_width) // 2, 0)
                top = max((size[1] - zoom_height) // 2, 0)
                zoomed = frame[top : top + zoom_height, left : left + zoom_width]
                zoomed = cv2.resize(zoomed, size, interpolation=cv2.INTER_CUBIC)

                if title_text:
                    overlay = zoomed.copy()
                    cv2.rectangle(overlay, (0, size[1] - 130), (size[0], size[1]), (255, 255, 255), -1)
                    zoomed = cv2.addWeighted(overlay, 0.82, zoomed, 0.18, 0)
                    display_title = title_text[:58]
                    cv2.putText(
                        zoomed,
                        display_title,
                        (48, size[1] - 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.05,
                        (25, 31, 38),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        zoomed,
                        "WAHA Marketplace",
                        (48, size[1] - 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (102, 112, 133),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(zoomed)
    finally:
        writer.release()

    return str(output)


def enhance_all_images(folder_path: str | Path) -> list[str]:
    folder = Path(folder_path)
    images = [
        item
        for item in sorted(folder.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return improve_images_batch(images, folder / "final_enhanced")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ameliorer des images produit style marketplace.")
    parser.add_argument("folder", help="Dossier contenant les images a ameliorer")
    args = parser.parse_args()

    enhance_all_images(args.folder)
