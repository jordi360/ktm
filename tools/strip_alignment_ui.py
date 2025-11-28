import argparse
import json
import os
from glob import glob

import cv2
import numpy as np


WINDOW_NAME = "Strip Alignment UI"


def find_images(folder, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    images = []
    for ext in exts:
        images.extend(glob(os.path.join(folder, f"*{ext}")))
    images.sort()
    return images


class StripAlignmentUI:
    def __init__(self, image_paths, overlap_fraction=0.15, alpha_preview=0.35):
        if not image_paths:
            raise ValueError("At least one image is required to start the editor.")

        self.paths = []
        self.images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: could not read {path}, skipping.")
                continue
            self.paths.append(path)
            self.images.append(img)

        if not self.images:
            raise ValueError("None of the provided paths could be loaded as images.")

        self.wrap_width = 0
        self._update_wrap_width(warn=True)

        # Initial positions stack vertically with slight overlap
        self.positions = []
        current_y = 0.0
        for img in self.images:
            self.positions.append([0.0, current_y])
            h = img.shape[0]
            step = h * (1.0 - overlap_fraction)
            current_y += max(25.0, step)

        self.active_index = 0
        self.dragging = False
        self.drag_offset = (0.0, 0.0)

        self.transparency_enabled = False
        self.transparency_alpha = alpha_preview

        self.last_canvas_shape = None
        self.last_shift = (0.0, 0.0)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

    def _update_wrap_width(self, warn=False):
        widths = [img.shape[1] for img in self.images]
        if not widths:
            self.wrap_width = 0
            return
        if warn and len(set(widths)) > 1:
            print("Warning: input strips have varying widths; wrapping uses the widest image for horizontal wrapping.")
        self.wrap_width = max(widths)

    def _wrap_value(self, value, period):
        if period <= 0:
            return value
        wrapped = value % period
        if wrapped < 0:
            wrapped += period
        return wrapped

    def _vertical_bounds(self):
        ys = []
        for img, (_, oy) in zip(self.images, self.positions):
            h = img.shape[0]
            ys.extend([oy, oy + h])
        min_y = min(ys)
        max_y = max(ys)
        return min_y, max_y

    def build_canvas(self, active_alpha_override=None):
        padding = 80
        wrap_width = max(1, int(round(self.wrap_width))) if self.wrap_width else max(
            1, self.images[0].shape[1]
        )
        min_y, max_y = self._vertical_bounds()
        span_y = max_y - min_y
        canvas_w = wrap_width + padding * 2 + 2
        canvas_h = max(1, int(np.ceil(span_y)) + padding * 2 + 2)
        shift_x = padding
        shift_y = -min_y + padding

        acc = np.zeros((canvas_h, canvas_w, 3), np.float32)
        weight = np.zeros((canvas_h, canvas_w, 1), np.float32)

        if active_alpha_override is None:
            active_alpha = self.transparency_alpha if self.transparency_enabled else 1.0
        else:
            active_alpha = active_alpha_override

        wrap_shifts = [-wrap_width, 0, wrap_width] if wrap_width > 0 else [0]

        for idx, (img, pos) in enumerate(zip(self.images, self.positions)):
            h, w = img.shape[:2]
            base_x = self._wrap_value(pos[0], self.wrap_width) if self.wrap_width else pos[0]
            base_y = pos[1]
            self.positions[idx][0] = base_x

            alpha = active_alpha if idx == self.active_index else 1.0
            for shift in wrap_shifts:
                draw_x = base_x + shift
                x0 = int(np.floor(draw_x + shift_x))
                y0 = int(np.floor(base_y + shift_y))
                x1 = x0 + w
                y1 = y0 + h

                if x1 <= 0 or y1 <= 0 or x0 >= canvas_w or y0 >= canvas_h:
                    continue

                clip_x0 = max(0, x0)
                clip_y0 = max(0, y0)
                clip_x1 = min(canvas_w, x1)
                clip_y1 = min(canvas_h, y1)
                if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
                    continue

                src_x0 = clip_x0 - x0
                src_x1 = src_x0 + (clip_x1 - clip_x0)
                src_y0 = clip_y0 - y0
                src_y1 = src_y0 + (clip_y1 - clip_y0)

                patch = img[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
                acc[clip_y0:clip_y1, clip_x0:clip_x1] += patch * alpha
                weight[clip_y0:clip_y1, clip_x0:clip_x1] += alpha

        mask = weight > 0
        canvas = np.zeros_like(acc, dtype=np.float32)
        np.divide(acc, weight, out=canvas, where=mask)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        # Draw overlays for clarity
        for idx, (img, pos) in enumerate(zip(self.images, self.positions)):
            h, w = img.shape[:2]
            base_x = pos[0]
            base_y = pos[1]
            color = (0, 200, 0) if idx != self.active_index else (0, 255, 255)
            thickness = 2 if idx == self.active_index else 1
            for shift in wrap_shifts:
                draw_x = base_x + shift
                x0 = int(round(draw_x + shift_x))
                y0 = int(round(base_y + shift_y))
                x1 = x0 + w
                y1 = y0 + h
                if x1 <= 0 or y1 <= 0 or x0 >= canvas_w or y0 >= canvas_h:
                    continue
                draw_x0 = max(0, min(canvas_w - 1, x0))
                draw_y0 = max(0, min(canvas_h - 1, y0))
                draw_x1 = max(0, min(canvas_w - 1, x1 - 1))
                draw_y1 = max(0, min(canvas_h - 1, y1 - 1))
                if draw_x1 <= draw_x0 or draw_y1 <= draw_y0:
                    continue
                cv2.rectangle(canvas, (draw_x0, draw_y0), (draw_x1, draw_y1), color, thickness)

            label_x = int(round(base_x + shift_x + 5))
            label_y = int(round(base_y + shift_y + 20))
            label_x = max(5, min(canvas_w - 5, label_x))
            label_y = max(20, min(canvas_h - 5, label_y))
            cv2.putText(
                canvas,
                f"{idx}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        basename = os.path.basename(self.paths[self.active_index])
        alpha_state = "ON" if self.transparency_enabled else "OFF"
        info_lines = [
            f"Active #{self.active_index}: {basename}",
            f"Transparency: {alpha_state} (alpha={self.transparency_alpha:.2f}) | toggle [A]",
            "Horizontal drag wraps around the strip width.",
            "[ / ] cycle strips   |   Click+drag to move   |   X/Delete removes   |   S save   |   Q quits",
        ]
        for idx, line in enumerate(info_lines):
            cv2.putText(
                canvas,
                line,
                (10, 25 + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        self.last_canvas_shape = (canvas_h, canvas_w)
        self.last_shift = (shift_x, shift_y)
        return canvas

    def _window_to_canvas(self, x, y):
        if not self.last_canvas_shape:
            return x, y
        if hasattr(cv2, "getWindowImageRect"):
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            except cv2.error:
                win_w, win_h = 0, 0
        else:
            win_w, win_h = 0, 0

        if win_w <= 0 or win_h <= 0:
            return x, y

        canvas_h, canvas_w = self.last_canvas_shape
        scale_x = canvas_w / float(win_w)
        scale_y = canvas_h / float(win_h)
        return int(x * scale_x), int(y * scale_y)

    def _canvas_to_world(self, cx, cy):
        shift_x, shift_y = self.last_shift
        return cx - shift_x, cy - shift_y

    def _index_at_world(self, wx, wy):
        wrapped_x = self._wrap_value(wx, self.wrap_width) if self.wrap_width else wx
        for idx in reversed(range(len(self.images))):
            img = self.images[idx]
            h, w = img.shape[:2]
            ox, oy = self.positions[idx]
            if ox <= wrapped_x < ox + w and oy <= wy < oy + h:
                return idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        canvas_x, canvas_y = self._window_to_canvas(x, y)
        world_x, world_y = self._canvas_to_world(canvas_x, canvas_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._index_at_world(world_x, world_y)
            if idx is None:
                self.dragging = False
                return
            self.active_index = idx
            self.dragging = True
            ox, oy = self.positions[idx]
            wrapped_x = self._wrap_value(world_x, self.wrap_width) if self.wrap_width else world_x
            self.drag_offset = (wrapped_x - ox, world_y - oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            wrapped_x = self._wrap_value(world_x, self.wrap_width) if self.wrap_width else world_x
            ox = wrapped_x - self.drag_offset[0]
            if self.wrap_width:
                ox = self._wrap_value(ox, self.wrap_width)
            oy = world_y - self.drag_offset[1]
            self.positions[self.active_index][0] = ox
            self.positions[self.active_index][1] = oy
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def cycle_active(self, direction):
        if not self.images:
            return
        self.active_index = (self.active_index + direction) % len(self.images)

    def remove_active(self):
        if not self.images:
            return
        removed = self.paths.pop(self.active_index)
        self.images.pop(self.active_index)
        self.positions.pop(self.active_index)
        print(f"Removed strip: {removed}")
        self._update_wrap_width()
        if not self.images:
            return
        self.active_index = min(self.active_index, len(self.images) - 1)

    def save_state(self, basename):
        if not basename:
            print("No output basename provided; skipping save.")
            return
        canvas = self.build_canvas(active_alpha_override=1.0)
        image_path = basename + "_aligned.png"
        cv2.imwrite(image_path, canvas)
        data = {"strips": []}
        for path, (ox, oy), img in zip(self.paths, self.positions, self.images):
            h, w = img.shape[:2]
            data["strips"].append(
                {
                    "path": path,
                    "filename": os.path.basename(path),
                    "offset_x": float(ox),
                    "offset_y": float(oy),
                    "width": int(w),
                    "height": int(h),
                }
            )
        json_path = basename + "_offsets.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved composed image to {image_path}")
        print(f"Saved offsets to {json_path}")

    def run(self, output_basename=None):
        while True:
            if not self.images:
                print("All strips were removed. Exiting.")
                break
            canvas = self.build_canvas()
            cv2.imshow(WINDOW_NAME, canvas)
            raw_key = cv2.waitKey(20)
            if raw_key == -1:
                key = None
            else:
                key = raw_key & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("["):
                self.cycle_active(-1)
            elif key == ord("]"):
                self.cycle_active(1)
            elif key == ord("a"):
                self.transparency_enabled = not self.transparency_enabled
            elif key == ord("s"):
                self.save_state(output_basename)
            elif key in (ord("x"), ord("d"), 8, 127):
                prev_count = len(self.images)
                self.remove_active()
                if not self.images:
                    break
                if len(self.images) != prev_count:
                    self.active_index = min(self.active_index, len(self.images) - 1)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Quick UI to align pre-unwrapped strips via drag-and-drop.")
    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        help="Folder containing pre-unwrapped strips that should be aligned.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Initial overlap guess used to stack strips vertically (default: 0.15).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Transparency applied to the active image while preview mode is enabled (default: 0.35).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional basename for saving the composed mosaic and offsets. Defaults to <folder>/<folder> if omitted.",
    )
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    images = find_images(folder)
    if not images:
        raise SystemExit("No images found in the provided folder.")

    try:
        ui = StripAlignmentUI(images, overlap_fraction=args.overlap, alpha_preview=args.alpha)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output:
        output_basename = args.output
    else:
        folder_name = os.path.basename(os.path.abspath(folder.rstrip("/")))
        output_basename = os.path.join(folder, folder_name)

    print("Controls:")
    print("  Click + drag        : move active strip (both axes).")
    print("  [ / ]               : cycle through strips.")
    print("  A                   : toggle transparency preview for the active strip.")
    print("  Horizontal drag     : wrap-around so exiting right re-enters from the left.")
    print("  X / Delete / D      : remove the active strip.")
    print("  S                   : save current composition and offsets.")
    print("  Q / ESC             : quit.")

    ui.run(output_basename=output_basename)


if __name__ == "__main__":
    main()
