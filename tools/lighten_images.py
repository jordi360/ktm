#!/usr/bin/env python3
"""Batch tool to lighten dark captures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageEnhance

# Extensions that will be picked up when scanning a directory.
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tiff",
    ".tif",
    ".webp",
}

# Pillow expects specific names for some output formats.
PIL_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "tif": "TIFF",
    "tiff": "TIFF",
    "bmp": "BMP",
    "gif": "GIF",
    "webp": "WEBP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lighten an image or every image inside a folder. "
            "The output is stored as the requested format (default: jpg)."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Image file or directory that contains images to process.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the lightened images will be written.",
    )
    parser.add_argument(
        "--extra",
        type=float,
        default=35.0,
        help=(
            "How much extra light to apply in percent. "
            "25 means 25%% brighter. Negative numbers darken the image. Default: 35."
        ),
    )
    parser.add_argument(
        "--output-format",
        default="jpg",
        help="Image format/extension to save. Default: jpg",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When the input is a directory, scan all nested folders as well.",
    )
    return parser.parse_args()


def gather_images(
    input_path: Path, recursive: bool
) -> Iterable[Tuple[Path, Path]]:
    """Yield (source_path, relative_path) pairs."""
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"{input_path} does not look like an image")
        yield input_path, Path(input_path.name)
        return

    if not input_path.is_dir():
        raise ValueError(f"{input_path} is not a file or directory")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    for entry in sorted(iterator):
        if entry.is_file() and is_image_file(entry):
            yield entry, entry.relative_to(input_path)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def compute_factor(extra_percent: float) -> float:
    return 1.0 + (extra_percent / 100.0)


def resolve_output_format(requested: str) -> Tuple[str, str]:
    ext = requested.lower().lstrip(".")
    pil_format = PIL_FORMATS.get(ext, ext.upper())
    return ext, pil_format


def ensure_output_path(base_dir: Path, relative: Path) -> Path:
    destination = base_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def lighten_image(src: Path, dst: Path, factor: float, pil_format: str) -> None:
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        enhancer = ImageEnhance.Brightness(rgb)
        brightened = enhancer.enhance(factor)
        brightened.save(dst, format=pil_format)


def main() -> int:
    args = parse_args()
    factor = compute_factor(args.extra)
    ext, pil_format = resolve_output_format(args.output_format)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        work_items = list(gather_images(args.input_path, args.recursive))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not work_items:
        print(f"No images found inside {args.input_path}", file=sys.stderr)
        return 1

    for src, relative in work_items:
        destination = ensure_output_path(output_dir, relative.with_suffix(f".{ext}"))
        lighten_image(src, destination, factor, pil_format)
        print(f"{src} -> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
