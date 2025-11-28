import os
import json
import math
import argparse

import cv2
import numpy as np


# ---------- Donut unwrap helpers ----------

def build_polar_remap(cx, cy, r_inner, r_outer, out_w=None, out_h=None):
    radial_extent = r_outer - r_inner

    if out_h is None:
        out_h = radial_extent

    if out_w is None:
        r_mean = 0.5 * (r_inner + r_outer)
        out_w = int(2 * math.pi * r_mean)

    v = np.linspace(0, 1, out_h, endpoint=False).astype(np.float32)
    u = np.linspace(0, 1, out_w, endpoint=False).astype(np.float32)

    theta = (u[None, :] * 2.0 * math.pi)     # shape (1, out_w)
    radius = r_inner + v[:, None] * radial_extent  # shape (out_h, 1)

    map_x = cx + radius * np.cos(theta)
    map_y = cy + radius * np.sin(theta)

    return map_x.astype(np.float32), map_y.astype(np.float32), out_w, out_h


def unwrap_donut(img, cx, cy, r_inner, r_outer, out_w=None, out_h=None):
    map_x, map_y, out_w, out_h = build_polar_remap(cx, cy, r_inner, r_outer, out_w, out_h)

    unwrapped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return unwrapped


# ---------- Mosaic builder ----------

def build_mosaic_backend(input_folder, params_json_path, offsets_json_path, output=None, feather=True):

    # --- Load donut params JSON ---
    with open(params_json_path, "r") as f:
        donut_params = json.load(f)

    # donut_params must contain:
    # {
    #   "images": [
    #       {
    #           "filename": "cam1.png",
    #           "center": [...],
    #           "inner_radius": ...,
    #           "outer_radius": ...
    #       },
    #       ...
    #   ]
    # }

    # --- Load offsets JSON from UI ---
    with open(offsets_json_path, "r") as f:
        offsets_data = json.load(f)

    offsets_map = {entry["filename"]: entry["offset_y"] for entry in offsets_data["cameras"]}

    strips = []
    offsets = []
    names = []

    print("Unwrapping all images...")

    for entry in donut_params["images"]:
        filename = entry["filename"]
        img_path = os.path.join(input_folder, filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        cx, cy = entry["center"]
        r_inner = entry["inner_radius"]
        r_outer = entry["outer_radius"]

        print(f"  - Unwrapping {filename}")

        strip = unwrap_donut(img, cx, cy, r_inner, r_outer)

        strips.append(strip)
        offsets.append(offsets_map[filename])
        names.append(filename)

    # --- Resize all strips to the maximum width ---
    max_w = max(s.shape[1] for s in strips)
    resized_strips = []
    heights = []

    for s in strips:
        h, w = s.shape[:2]
        if w != max_w:
            scale = max_w / w
            new_h = int(h * scale)
            s_resized = cv2.resize(s, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_strips.append(s_resized)
            heights.append(new_h)
        else:
            resized_strips.append(s)
            heights.append(h)

    # --- Compute canvas size ---
    min_y = min(offsets)
    max_y = max(off + h for off, h in zip(offsets, heights))

    shift = -min_y if min_y < 0 else 0
    canvas_h = max_y + shift
    canvas_w = max_w

    print(f"Canvas size: {canvas_w}×{canvas_h}   (shift={shift})")

    acc = np.zeros((canvas_h, canvas_w, 3), np.float32)
    weight = np.zeros((canvas_h, canvas_w), np.float32)

    # --- Blending ---
    for img, off in zip(resized_strips, offsets):
        h, w = img.shape[:2]
        y0 = off + shift
        y1 = y0 + h

        roi_acc = acc[y0:y1, :w]
        roi_weight = weight[y0:y1, :w]

        if feather:
            y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            band = np.minimum(y, 1 - y) * 2.0
            strip_weight = band
        else:
            strip_weight = np.ones((h, 1), np.float32)

        roi_acc += img.astype(np.float32) * strip_weight
        roi_weight += strip_weight

    canvas = np.zeros_like(acc, dtype=np.uint8)
    valid = weight > 0
    canvas[valid] = (acc[valid] / weight[valid, None]).astype(np.uint8)

    # --- Save output ---
    if output is None:
        base, _ = os.path.splitext(offsets_json_path)
        output = base + "_mosaic.png"

    cv2.imwrite(output, canvas)
    print(f"Saved mosaic: {output}")

    return canvas


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Backend donut unwrap + mosaic blending")

    parser.add_argument("--input", required=True,
                        help="Folder with circular images")
    parser.add_argument("--params", required=True,
                        help="Donut params JSON file")
    parser.add_argument("--offsets", required=True,
                        help="Offsets JSON from mosaic UI")
    parser.add_argument("--output", "-o", default=None,
                        help="Output mosaic path")
    parser.add_argument("--no-feather", action="store_true",
                        help="Disable feather blending")

    args = parser.parse_args()

    build_mosaic_backend(
        input_folder=args.input,
        params_json_path=args.params,
        offsets_json_path=args.offsets,
        output=args.output,
        feather=not args.no_feather
    )


if __name__ == "__main__":
    main()


'''

This code gets a folder of the full capture and initially unwrap it all the donut images and then stitch them into a mosaic based on offsets specified in a JSON file (produced by the donut_mosaic_ui.py tool).
that will generate the expected complete unwrapped image of the innerpart of the object.

python donut_mosaic_backend.py \
    --input data/ \
    --params data/donut_params.json \
    --offsets data/mosaic_offsets.json \
    --output final_mosaic.png

python donut_mosaic_backend.py \
    --input data/ \
    --params data/donut_params.json \
    --offsets data/mosaic_offsets.json \
    --no-feather
    
'''