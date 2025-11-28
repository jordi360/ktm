import os
import json
import math
import argparse
from glob import glob

import cv2
import numpy as np


WINDOW_NAME = "Donut Mosaic Editor"


# ---------- Donut unwrap helpers (from previous logic) ----------

def load_donut_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cx, cy = data["center"]
    r_inner = data["inner_radius"]
    r_outer = data["outer_radius"]
    image_path = data.get("image_path", None)
    return image_path, cx, cy, r_inner, r_outer


def build_polar_remap(cx, cy, r_inner, r_outer, out_w=None, out_h=None):
    radial_extent = r_outer - r_inner

    if out_h is None:
        out_h = radial_extent

    if out_w is None:
        r_mean = 0.5 * (r_inner + r_outer)
        out_w = int(2 * math.pi * r_mean)

    v = np.linspace(0, 1, out_h, endpoint=False).astype(np.float32)
    u = np.linspace(0, 1, out_w, endpoint=False).astype(np.float32)

    theta = (u[None, :] * 2.0 * math.pi)
    radius = r_inner + v[:, None] * radial_extent

    map_x = cx + radius * np.cos(theta)
    map_y = cy + radius * np.sin(theta)

    return map_x.astype(np.float32), map_y.astype(np.float32), out_w, out_h


def unwrap_donut_in_memory(image_path, json_path, out_w=None, out_h=None, use_json_image_path=True):
    json_image_path, cx, cy, r_inner, r_outer = load_donut_params(json_path)
    if use_json_image_path and json_image_path and os.path.isfile(json_image_path):
        image_path = json_image_path

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    map_x, map_y, out_w, out_h = build_polar_remap(cx, cy, r_inner, r_outer, out_w, out_h)

    unwrapped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return unwrapped


# ---------- Mosaic editor ----------

class MosaicEditor:
    def __init__(self, strips, filenames, overlap_fraction=0.2):
        """
        strips: list of HxWx3 uint8 unwrapped images (all same width)
        filenames: list of original image filenames (same length as strips)
        overlap_fraction: initial guess of vertical overlap between strips
        """
        assert len(strips) == len(filenames)
        self.strips = strips
        self.filenames = filenames
        self.n = len(strips)
        self.active_index = 0

        self.width = strips[0].shape[1]
        self.heights = [s.shape[0] for s in strips]

        # initial vertical offsets: stack with some overlap
        self.offsets = []
        current_y = 0
        for h in self.heights:
            self.offsets.append(int(current_y))
            step = int(h * (1.0 - overlap_fraction))
            current_y += max(1, step)

        # drag state
        self.dragging = False
        self.drag_start_y = 0
        self.offset_start = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

    def build_mosaic(self):
        # compute canvas bounds
        min_y = min(off for off in self.offsets)
        max_y = max(off + h for off, h in zip(self.offsets, self.heights))

        # shift so min_y >= 0
        shift = -min_y if min_y < 0 else 0
        canvas_h = max_y + shift
        canvas_w = self.width

        acc = np.zeros((canvas_h, canvas_w, 3), np.float32)
        weight = np.zeros((canvas_h, canvas_w, 1), np.float32)

        for img, off in zip(self.strips, self.offsets):
            h, w = img.shape[:2]
            y0 = off + shift
            y1 = y0 + h

            # blending via simple averaging
            acc[y0:y1, 0:w] += img.astype(np.float32)
            weight[y0:y1, 0:w] += 1.0

        # avoid division by zero
        mask = weight > 0
        canvas = np.zeros_like(acc, dtype=np.uint8)
        canvas[mask.squeeze(-1)] = (acc[mask] / weight[mask]).astype(np.uint8)

        # visual indicators: draw strip indices on rough centers
        for idx, (off, h) in enumerate(zip(self.offsets, self.heights)):
            y_center = off + shift + h // 2
            if 0 <= y_center < canvas_h:
                cv2.putText(
                    canvas,
                    f"{idx}",
                    (10, int(y_center)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # active strip indicator
        cv2.putText(
            canvas,
            f"Active strip: {self.active_index} ({os.path.basename(self.filenames[self.active_index])})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Use [ / ] to change strip, drag with mouse to move, 's' to save, 'q'/ESC to quit.",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas, shift, canvas_h, canvas_w

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_y = y
            self.offset_start = self.offsets[self.active_index]

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dy = y - self.drag_start_y
            self.offsets[self.active_index] = self.offset_start + dy

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def run(self, output_basename):
        while True:
            mosaic, shift, canvas_h, canvas_w = self.build_mosaic()
            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('['):
                self.active_index = (self.active_index - 1) % self.n
            elif key == ord(']'):
                self.active_index = (self.active_index + 1) % self.n
            elif key == ord('s'):
                # save mosaic
                out_img_path = output_basename + "_mosaic.png"
                cv2.imwrite(out_img_path, mosaic)
                print(f"Saved mosaic image to {out_img_path}")

                # save JSON with offsets (in unshifted coordinates)
                data = {
                    "canvas_width": int(canvas_w),
                    "canvas_height": int(canvas_h),
                    "shift_applied": int(shift),
                    "cameras": []
                }
                for fname, off, h in zip(self.filenames, self.offsets, self.heights):
                    entry = {
                        "filename": os.path.basename(fname),
                        "offset_y": int(off),  # before shift
                        "height": int(h)
                    }
                    data["cameras"].append(entry)

                out_json_path = output_basename + "_mosaic_offsets.json"
                with open(out_json_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved mosaic offsets to {out_json_path}")

        cv2.destroyAllWindows()


# ---------- Main pipeline: folder → unwrap → UI ----------

def find_images(folder, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    images = []
    for ext in exts:
        images.extend(glob(os.path.join(folder, f"*{ext}")))
    images.sort()
    return images


def main():
    parser = argparse.ArgumentParser(description="UI to align and blend unwrapped donut images vertically.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Folder containing the donut images to unwrap",
    )
    parser.add_argument(
        "--params",
        "-p",
        required=True,
        help="Path to JSON file with donut parameters (center/radii)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Initial guess of vertical overlap fraction between strips (default=0.2)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        return

    if not os.path.isfile(args.params):
        print(f"Params JSON not found: {args.params}")
        return

    images = find_images(args.input)
    if not images:
        print("No images found in folder.")
        return

    print(f"Found {len(images)} images to unwrap and stitch using {args.params}.")

    # unwrap all strips
    strips = []
    for img_path in images:
        print(f"Unwrapping {img_path}...")
        strip = unwrap_donut_in_memory(img_path, args.params, use_json_image_path=False)
        strips.append(strip)

    # unify widths
    max_w = max(s.shape[1] for s in strips)
    resized_strips = []
    for s in strips:
        h, w = s.shape[:2]
        if w != max_w:
            new_h = int(h * max_w / w)
            s_resized = cv2.resize(s, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_strips.append(s_resized)
        else:
            resized_strips.append(s)

    editor = MosaicEditor(resized_strips, images, overlap_fraction=args.overlap)

    folder_name = os.path.basename(os.path.abspath(args.input.rstrip("/")))
    output_basename = os.path.join(args.input, folder_name)
    editor.run(output_basename)


if __name__ == "__main__":
    main()
