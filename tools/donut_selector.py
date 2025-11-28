import cv2
import json
import os
import numpy as np


# ====== CONFIG ======
WINDOW_NAME = "Donut Circle Selector"
CLICK_TOLERANCE = 10  # pixels distance to decide if you're "on" a circle or center


class DonutSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        h, w = self.image.shape[:2]

        # Initial guess: center of image
        self.cx = w // 2
        self.cy = h // 2

        # Initial radii based on image size
        base = min(w, h)
        self.r_inner = base // 8
        self.r_outer = base // 4

        # Interaction state
        self.drag_mode = None  # 'center', 'inner', 'outer', or None

    def draw_overlay(self):
        """Draw circles and crosshair over the image and return a copy."""
        vis = self.image.copy()

        # Draw outer and inner circles
        cv2.circle(vis, (self.cx, self.cy), self.r_outer, (0, 0, 255), 2)  # red outer
        cv2.circle(vis, (self.cx, self.cy), self.r_inner, (0, 255, 0), 2)  # green inner

        # Crosshair at center
        cross_len = 20
        cv2.line(vis, (self.cx - cross_len, self.cy), (self.cx + cross_len, self.cy), (255, 255, 255), 1)
        cv2.line(vis, (self.cx, self.cy - cross_len), (self.cx, self.cy + cross_len), (255, 255, 255), 1)
        cv2.circle(vis, (self.cx, self.cy), 3, (255, 255, 255), -1)

        # Instructions
        cv2.putText(
            vis,
            "Left-drag center or circles. Press 's' to save JSON, 'q'/ESC to quit.",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return vis

    def mouse_callback(self, event, x, y, flags, param):
        # Distance from click to center
        dx = x - self.cx
        dy = y - self.cy
        dist = np.hypot(dx, dy)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Decide what we are going to drag based on where we clicked
            if dist < CLICK_TOLERANCE:
                self.drag_mode = "center"
            elif abs(dist - self.r_inner) < CLICK_TOLERANCE:
                self.drag_mode = "inner"
            elif abs(dist - self.r_outer) < CLICK_TOLERANCE:
                self.drag_mode = "outer"
            else:
                self.drag_mode = None

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.drag_mode == "center":
                self.cx = x
                self.cy = y
            elif self.drag_mode == "inner":
                # Radius = distance from center to mouse
                new_r = int(dist)
                new_r = max(1, min(new_r, self.r_outer - 1))
                self.r_inner = new_r
            elif self.drag_mode == "outer":
                new_r = int(dist)
                new_r = max(self.r_inner + 1, new_r)
                self.r_outer = new_r

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None

    def save_json(self):
        h, w = self.image.shape[:2]
        data = {
            "image_path": self.image_path,
            "image_width": w,
            "image_height": h,
            "center": [int(self.cx), int(self.cy)],
            "inner_radius": int(self.r_inner),
            "outer_radius": int(self.r_outer),
        }

        base, ext = os.path.splitext(self.image_path)
        json_path = base + "_donut_params.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved parameters to {json_path}")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            vis = self.draw_overlay()
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s"):
                self.save_json()
            elif key == ord("q") or key == 27:  # 'q' or ESC
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change this path to your image file, or adapt to use argparse
    # image_path = "/home/jordi/Documents/ais/ktm/boroscopica/E151/metal_nok/3.png"
    image_path = "/home/jordi/Documents/ais/ktm/captures/E198/metal/26.png"
    selector = DonutSelector(image_path)
    selector.run()
