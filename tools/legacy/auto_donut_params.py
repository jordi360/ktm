import argparse
import json
import math
import os

import cv2
import numpy as np


def find_concentric_circles(circles, center_tolerance=20):
    """
    Find pairs of concentric circles (circles with same center within tolerance).
    Returns the pair that likely represents the donut (inner + outer).
    """
    if circles is None or len(circles[0]) < 2:
        return None

    c = circles[0]
    n = len(c)

    # Find all concentric pairs
    concentric_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, r1 = c[i]
            cx2, cy2, r2 = c[j]

            # Check if centers are close (concentric)
            center_dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if center_dist < center_tolerance:
                # Store as (center_x, center_y, inner_radius, outer_radius)
                if r1 < r2:
                    concentric_pairs.append((cx1, cy1, r1, r2, center_dist))
                else:
                    concentric_pairs.append((cx1, cy1, r2, r1, center_dist))

    if not concentric_pairs:
        return None

    # Choose the pair with best concentricity (smallest center distance)
    # and reasonable radius ratio (donut should have outer > 1.3 * inner)
    best_pair = None
    best_score = float('inf')

    for cx, cy, r_inner, r_outer, center_dist in concentric_pairs:
        ratio = r_outer / r_inner if r_inner > 0 else 0
        # Good donut ratio is typically between 1.3 and 3.0
        if 1.3 <= ratio <= 3.0:
            # Score: prefer pairs with closer centers and good ratio
            score = center_dist + abs(ratio - 1.8) * 10  # penalize ratios far from ideal 1.8
            if score < best_score:
                best_score = score
                best_pair = (cx, cy, r_inner, r_outer)

    return best_pair


def estimate_center(img):
    """
    Estimate donut center by finding concentric circles (inner + outer of donut).
    This filters out the third tangent circle that has a different center.
    Fallback: threshold centroid or image center.
    Returns: (cx, cy, r_inner, r_outer, all_circles_for_debug)
    """
    h, w = img.shape
    blur = cv2.medianBlur(img, 5)

    # Try multiple parameter sets to detect circles reliably
    param_sets = [
        {'dp': 1.2, 'minDist': 80, 'param1': 100, 'param2': 30},
        {'dp': 1.0, 'minDist': 60, 'param1': 80, 'param2': 25},
        {'dp': 1.5, 'minDist': 100, 'param1': 120, 'param2': 35},
    ]

    all_circles_combined = None
    for params in param_sets:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['minDist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=30,
            maxRadius=int(min(h, w) / 2),
        )
        if circles is not None:
            # Keep the first successful detection for visualization
            if all_circles_combined is None:
                all_circles_combined = circles

            concentric_pair = find_concentric_circles(circles, center_tolerance=20)
            if concentric_pair is not None:
                cx, cy, r_inner, r_outer = concentric_pair
                print(f"Found concentric donut circles: center=({cx:.1f}, {cy:.1f}), "
                      f"inner_r={r_inner:.1f}, outer_r={r_outer:.1f}")
                return cx, cy, r_inner, r_outer, circles

    # Fallback: centroid of Otsu-thresholded bright region
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    M = cv2.moments(th)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = w / 2.0, h / 2.0

    return cx, cy, None, None, all_circles_combined


def radial_profile(img, cx, cy, max_r=None, n_theta=720):
    """
    Angular mean intensity as function of radius around (cx, cy).
    """
    h, w = img.shape
    if max_r is None:
        max_r = int(min(h, w) / 2) - 2

    thetas = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    rs = np.arange(0, max_r + 1, dtype=np.int32)
    profile = np.zeros_like(rs, dtype=np.float32)

    for i, r in enumerate(rs):
        xs = cx + r * np.cos(thetas)
        ys = cy + r * np.sin(thetas)

        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_i = xs[mask].astype(np.int32)
        ys_i = ys[mask].astype(np.int32)

        if xs_i.size == 0:
            profile[i] = 0.0
        else:
            profile[i] = img[ys_i, xs_i].mean()

    return rs, profile


def find_donut_radii(rs, profile,
                     inner_level=0.5,
                     outer_level=0.5):
    """
    NEW: half-maximum based radii detection.

    1. Smooth radial profile.
    2. Find main bright peak (donut ring).
    3. Inner radius = where curve rises above
       I_inner_thr = I_hole + inner_level * (I_peak - I_hole)
    4. Outer radius = where curve falls below
       I_outer_thr = I_bg + outer_level * (I_peak - I_bg)

    I_hole  = median intensity at small radii (inner dark area)
    I_bg    = median intensity at large radii (outer background)

    This is much more robust than pure gradient for your case
    and naturally ignores the outer partial “ghost” ring.
    """
    max_r = rs[-1]

    # Smooth profile
    arr = profile.reshape(1, -1).astype(np.float32)
    smooth = cv2.GaussianBlur(arr, (0, 0), sigmaX=5).ravel()

    # --- 1) main peak (donut band) ---
    r_min_peak = int(max_r * 0.05)
    r_max_peak = int(max_r * 0.6)  # allow up to 60% of radius
    peak_mask = (rs >= r_min_peak) & (rs <= r_max_peak)
    r_peak = rs[peak_mask][np.argmax(smooth[peak_mask])]
    I_peak = smooth[rs == r_peak][0]

    # --- 2) reference levels ---
    hole_mask = rs < int(max_r * 0.05)
    I_hole = np.median(smooth[hole_mask])

    bg_mask = rs > int(max_r * 0.7)
    I_bg = np.median(smooth[bg_mask])

    # --- 3) inner radius (rising edge) ---
    inner_thr = I_hole + inner_level * (I_peak - I_hole)
    inner_candidates = rs[(rs > 0) & (rs < r_peak) & (smooth > inner_thr)]
    if len(inner_candidates) > 0:
        r_inner = inner_candidates[0]
    else:
        r_inner = int(0.7 * r_peak)

    # --- 4) outer radius (falling edge) ---
    outer_thr = I_bg + outer_level * (I_peak - I_bg)
    outer_candidates = rs[(rs > r_peak) & (smooth < outer_thr)]
    if len(outer_candidates) > 0:
        r_outer = outer_candidates[0]
    else:
        r_outer = int(1.3 * r_peak)

    return int(r_inner), int(r_outer), int(r_peak)


def save_params_json(image_path, cx, cy, r_inner, r_outer, out_json=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    data = {
        "image_path": os.path.abspath(image_path),
        "image_width": int(w),
        "image_height": int(h),
        "center": [int(round(cx)), int(round(cy))],
        "inner_radius": int(round(r_inner)),
        "outer_radius": int(round(r_outer)),
    }

    if out_json is None:
        base, _ = os.path.splitext(image_path)
        out_json = base + "_donut_params.json"

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved params JSON to {out_json}")
    return out_json


def save_debug_image(image_path, cx, cy, r_inner, r_outer, out_debug=None, all_circles=None):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cx_i, cy_i = int(round(cx)), int(round(cy))

    # Draw all detected circles in yellow (rejected/other circles)
    if all_circles is not None:
        for circle in all_circles[0]:
            c_cx, c_cy, c_r = circle
            c_cx_i, c_cy_i, c_r_i = int(round(c_cx)), int(round(c_cy)), int(round(c_r))
            cv2.circle(color, (c_cx_i, c_cy_i), c_r_i, (0, 255, 255), 1)  # Yellow - all circles
            cv2.circle(color, (c_cx_i, c_cy_i), 3, (0, 255, 255), -1)  # Yellow center dot

    # Draw selected donut circles (will overdraw the yellow circles)
    # outer (red) + inner (green)
    cv2.circle(color, (cx_i, cy_i), int(r_outer), (0, 0, 255), 3)  # Red - outer donut
    cv2.circle(color, (cx_i, cy_i), int(r_inner), (0, 255, 0), 3)  # Green - inner donut

    # center crosshair (white)
    cross = 25
    cv2.line(color, (cx_i - cross, cy_i), (cx_i + cross, cy_i), (255, 255, 255), 2)
    cv2.line(color, (cx_i, cy_i - cross), (cx_i, cy_i + cross), (255, 255, 255), 2)
    cv2.circle(color, (cx_i, cy_i), 5, (255, 255, 255), -1)

    # Add legend
    legend_y = 30
    cv2.putText(color, "Green: Inner donut", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(color, "Red: Outer donut", (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(color, "Yellow: Other circles", (10, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if out_debug is None:
        base, _ = os.path.splitext(image_path)
        out_debug = base + "_donut_debug.png"

    cv2.imwrite(out_debug, color)
    print(f"Saved debug overlay to {out_debug}")
    return out_debug


def process_image(image_path, out_json=None, out_debug=None, max_r=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1) center and initial radius estimates from concentric circle detection
    cx, cy, r_inner_hint, r_outer_hint, all_circles = estimate_center(img)
    print(f"Estimated center: ({cx:.1f}, {cy:.1f})")

    # 2) If we got good radius hints from circle detection, use them directly
    if r_inner_hint is not None and r_outer_hint is not None:
        r_inner = int(round(r_inner_hint))
        r_outer = int(round(r_outer_hint))
        print(f"Using circle detection radii: inner={r_inner}, outer={r_outer}")
    else:
        # Fallback: use radial profile method
        print("Circle detection failed, using radial profile method")
        rs, prof = radial_profile(img, cx, cy, max_r=max_r)
        r_inner, r_outer, r_peak = find_donut_radii(rs, prof)
        print(f"Detected radii from profile: inner={r_inner}, outer={r_outer}, peak={r_peak}")

    # 3) save outputs with all circles for debug visualization
    json_path = save_params_json(image_path, cx, cy, r_inner, r_outer, out_json)
    debug_path = save_debug_image(image_path, cx, cy, r_inner, r_outer, out_debug, all_circles)

    return json_path, debug_path


def main():
    parser = argparse.ArgumentParser(
        description="Automatically detect concentric donut circles and generate params.json"
    )
    parser.add_argument("image", help="Input image file")
    parser.add_argument("--output-json", "-j", help="Output params.json path", default=None)
    parser.add_argument("--debug-image", "-d", help="Output debug PNG with circles", default=None)
    parser.add_argument(
        "--max-radius",
        type=int,
        default=None,
        help="Max radius for radial profile (default: min(H,W)/2)",
    )

    args = parser.parse_args()
    process_image(args.image, args.output_json, args.debug_image, args.max_radius)


if __name__ == "__main__":
    main()


'''
python auto_donut_params.py /home/jordi/Documents/ais/ktm/captures/E198/metal/17.png -j params.json -d a.png

'''