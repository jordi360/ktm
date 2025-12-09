#!/usr/bin/env python3
"""
Blend already-unwrapped cylindrical strips according to a JSON alignment file.

Typical usage:
    python tools/blend_unwrapped_strips.py \
        --images captures/E198/metal/unwrapped \
        --alignment captures/alignment.json \
        --output captures/E198/metal/unwrapped_blend_2.png \
        --central-fraction 0.55 --min-gap 80 --feather-px 50
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def positive_fraction(value: str) -> float:
    try:
        value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("fraction must be within (0, 1].")
    return value


def non_negative(value: str) -> float:
    try:
        value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0.")
    return value


def _central_crop(img: np.ndarray, fraction: float) -> Tuple[np.ndarray, int]:
    """Return central vertical slice and number of rows trimmed from the top."""
    if fraction >= 0.999:
        return img.copy(), 0
    h = img.shape[0]
    crop_h = max(1, int(round(h * fraction)))
    crop_h = min(crop_h, h)
    start = max(0, (h - crop_h) // 2)
    end = start + crop_h
    return img[start:end].copy(), start


def _build_row_weights(height: int, feather_px: int, enabled: bool) -> np.ndarray:
    weights = np.ones((height, 1), dtype=np.float32)
    if not enabled or feather_px <= 0:
        return weights

    fade = min(feather_px, height // 2)
    if fade <= 0:
        return weights

    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    weights[:fade, 0] = ramp
    weights[-fade:, 0] = ramp[::-1]
    return weights


def _blend_strip(
    acc: np.ndarray,
    weight: np.ndarray,
    strip: np.ndarray,
    row_weight: np.ndarray,
    x: float,
    y: float,
    wrap_width: Optional[int],
):
    """Add a strip to the accumulation buffers, handling horizontal wrapping."""
    h, w = strip.shape[:2]
    y0 = int(round(y))
    y1 = y0 + h
    canvas_h, canvas_w = acc.shape[:2]
    if y0 < 0 or y1 > canvas_h:
        raise ValueError(f"Strip y-range {y0}:{y1} outside canvas height {canvas_h}")

    base_x = int(round(x))
    segments = []

    if wrap_width and wrap_width > 0:
        base_x %= wrap_width
        remaining = w
        src_start = 0
        draw_x = base_x
        while remaining > 0:
            span = wrap_width - draw_x
            take = min(remaining, span)
            if take > 0:
                segments.append((src_start, draw_x, take))
                remaining -= take
                src_start += take
            draw_x = 0
    else:
        segments.append((0, base_x, w))

    strip_f = strip.astype(np.float32)
    for src_start, draw_x, take in segments:
        if draw_x < 0 or draw_x + take > canvas_w:
            raise ValueError("Strip extends beyond canvas width; set wrapWidth or enable wrapping.")
        roi_acc = acc[y0:y1, draw_x:draw_x + take]
        roi_weight = weight[y0:y1, draw_x:draw_x + take]
        roi_strip = strip_f[:, src_start:src_start + take]
        weight_color = row_weight[:, :, None] if row_weight.ndim == 2 else row_weight
        roi_acc += roi_strip * weight_color
        roi_weight += row_weight


def build_canvas(
    images_dir: str,
    alignment_path: str,
    output_path: str,
    central_fraction: float,
    min_gap: float,
    feather_px: int,
    max_count: Optional[int],
    wrap_width: Optional[int],
    padding: int,
    feather_enabled: bool,
):
    with open(alignment_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    entries = alignment.get("images", [])
    if not entries:
        raise ValueError("Alignment file does not contain any images.")

    wrap_width = wrap_width or alignment.get("wrapWidth")

    entries = sorted(entries, key=lambda e: e.get("y", 0))
    prepared: List[Dict] = []
    last_center = None

    for entry in entries:
        filename = entry.get("filename")
        if not filename:
            continue
        path = os.path.join(images_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Could not read {path}, skipping.")
            continue

        cropped, top_trim = _central_crop(img, central_fraction)
        effective_y = float(entry.get("y", 0)) + top_trim
        center_y = effective_y + cropped.shape[0] * 0.5

        if min_gap > 0.0 and last_center is not None:
            if center_y - last_center < min_gap:
                print(f"[info] Skipping {filename} (gap {center_y - last_center:.1f} < {min_gap}).")
                continue

        row_weight = _build_row_weights(cropped.shape[0], feather_px, feather_enabled)
        prepared.append(
            {
                "filename": filename,
                "image": cropped,
                "x": float(entry.get("x", 0.0)),
                "y": effective_y,
                "center": center_y,
                "row_weight": row_weight,
            }
        )
        last_center = center_y

        if max_count is not None and len(prepared) >= max_count:
            break

    if not prepared:
        raise RuntimeError("No strips left after filtering. Check the alignment data or filters.")

    if not wrap_width:
        wrap_width = max(item["image"].shape[1] for item in prepared)
        print(f"[info] wrapWidth missing in JSON; using widest strip ({wrap_width}px).")

    wrap_width = int(round(wrap_width))
    if wrap_width <= 0:
        raise ValueError("wrapWidth must be positive.")

    min_y = min(item["y"] for item in prepared) - padding
    max_y = max(item["y"] + item["image"].shape[0] for item in prepared) + padding
    span_y = max_y - min_y
    canvas_h = max(1, int(math.ceil(span_y)))
    shift = -min_y

    acc = np.zeros((canvas_h, wrap_width, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, wrap_width), dtype=np.float32)

    print(f"[info] Blending {len(prepared)} strips into {wrap_width}x{canvas_h} canvas.")

    for item in prepared:
        y_pos = item["y"] + shift
        _blend_strip(
            acc,
            weight,
            item["image"],
            item["row_weight"],
            item["x"],
            y_pos,
            wrap_width,
        )

    mask = weight > 1e-6
    canvas = np.zeros_like(acc, dtype=np.uint8)
    if np.any(mask):
        blended = acc[mask] / weight[mask][:, None]
        canvas[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[info] Saved blended unwrap to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend already-unwrapped strips using alignment offsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, help="Folder containing unwrap_#.png images.")
    parser.add_argument("--alignment", required=True, help="alignment.json path.")
    parser.add_argument("--output", required=True, help="Output mosaic path.")
    parser.add_argument(
        "--central-fraction",
        type=positive_fraction,
        default=0.55,
        help="Fraction of each strip's height to keep (centered).",
    )
    parser.add_argument(
        "--min-gap",
        type=non_negative,
        default=0.0,
        help="Skip strips whose centers are closer than this many pixels vertically.",
    )
    parser.add_argument(
        "--feather-px",
        type=int,
        default=25,
        help="Pixels to feather at top/bottom of each strip (0 disables).",
    )
    parser.add_argument(
        "--max-count", type=int, default=None, help="Optional maximum number of strips to use."
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=None,
        help="Override wrapWidth from JSON (pixels).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Extra blank pixels above/below the tallest strips.",
    )
    parser.add_argument(
        "--no-feather",
        action="store_true",
        help="Disable feather blending along the strip height.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_canvas(
        images_dir=args.images,
        alignment_path=args.alignment,
        output_path=args.output,
        central_fraction=args.central_fraction,
        min_gap=args.min_gap,
        feather_px=max(0, args.feather_px),
        max_count=args.max_count,
        wrap_width=args.wrap_width,
        padding=max(0, args.padding),
        feather_enabled=not args.no_feather,
    )


if __name__ == "__main__":
    main()
