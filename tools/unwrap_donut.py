import argparse
import json
import math
import os

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_VERTICAL_FOV_DEG = 65.0  # documented in lens.md for the PCBPN013-WG probe


def expand_path(path):
    return os.path.expanduser(path) if path else None


def prefixed_filename(filename, prefix="unwrap_"):
    if filename.startswith(prefix):
        return filename
    return f"{prefix}{filename}"


def default_output_path(image_path):
    directory = os.path.dirname(image_path) or "."
    filename = os.path.basename(image_path)
    return os.path.join(directory, prefixed_filename(filename))


def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)


def list_images_in_directory(directory):
    images = []
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        _, ext = os.path.splitext(entry)
        if ext.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(full_path)
    return images


def looks_like_directory(path):
    if path is None:
        return False
    if os.path.isdir(path):
        return True
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)):
        return True
    trimmed = path.rstrip(os.sep)
    if os.altsep:
        trimmed = trimmed.rstrip(os.altsep)
    base_name = os.path.basename(trimmed)
    _, ext = os.path.splitext(base_name)
    return ext == ""


def load_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cx, cy = data["center"]
    r_inner = data["inner_radius"]
    r_outer = data["outer_radius"]
    image_path = data.get("image_path", None)
    return image_path, cx, cy, r_inner, r_outer


def compute_output_dimensions(r_inner, r_outer, width_override=None, height_override=None, vertical_fov_deg=None):
    """
    Compute the unwrapped image dimensions.

    Width follows the circumference at the mean radius (unless overridden).
    Height can be deduced from the camera vertical FOV so the resulting
    panorama matches the physical proportions of the cylinder wall.  The
    derivation assumes the donut image captures a full 360° horizontally.
    """
    radial_extent = max(1, int(round(r_outer - r_inner)))

    if width_override is None:
        r_mean = 0.5 * (r_inner + r_outer)
        width = max(1, int(round(2 * math.pi * r_mean)))
    else:
        width = int(width_override)

    if height_override is not None:
        height = int(height_override)
    elif vertical_fov_deg is not None and vertical_fov_deg > 0:
        vertical_fov_rad = math.radians(vertical_fov_deg)
        # width ~= 2*pi*R  ->  R ~= width / (2*pi)
        # vertical span on the cylinder ~= 2*R*tan(v_fov/2)
        # combine both to express height in pixels directly from width
        height = max(1, int(round(width * math.tan(vertical_fov_rad / 2.0) / math.pi)))
    else:
        height = radial_extent

    return width, height


def build_polar_remap(cx, cy, r_inner, r_outer, out_w=None, out_h=None):
    """
    Build OpenCV remap maps to unwrap an annulus to a rectangle.
    Height corresponds to radius, width corresponds to angle.
    """
    radial_extent = r_outer - r_inner

    if out_h is None:
        out_h = radial_extent  # roughly 1 px per radial pixel

    if out_w is None:
        # use mean radius to set angular sampling ~1 px per pixel along circumference
        r_mean = 0.5 * (r_inner + r_outer)
        out_w = int(2 * math.pi * r_mean)

    # Create grid of output coordinates
    # v: [0..out_h-1] (rows) -> radius
    # u: [0..out_w-1] (cols) -> angle
    v = np.linspace(0, 1, out_h, endpoint=False).astype(np.float32)
    u = np.linspace(0, 1, out_w, endpoint=False).astype(np.float32)

    # Broadcast to 2D
    theta = (u[None, :] * 2.0 * math.pi)  # shape (1, out_w)
    radius = r_inner + v[:, None] * radial_extent  # shape (out_h, 1)

    # Compute source coordinates
    map_x = cx + radius * np.cos(theta)
    map_y = cy + radius * np.sin(theta)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return map_x, map_y, out_w, out_h


def unwrap_donut(
    image_path,
    json_path,
    output_path=None,
    out_w=None,
    out_h=None,
    vertical_fov_deg=None,
    show_result=True,
    preloaded_params=None,
):
    params = preloaded_params or load_params(json_path)
    json_image_path, cx, cy, r_inner, r_outer = params

    if image_path is None:
        candidate_path = json_image_path
    else:
        candidate_path = image_path

    if candidate_path is None:
        raise ValueError("No valid image_path provided in JSON or arguments.")

    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(f"Image does not exist: {candidate_path}")

    img = cv2.imread(candidate_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {candidate_path}")

    out_w, out_h = compute_output_dimensions(
        r_inner,
        r_outer,
        width_override=out_w,
        height_override=out_h,
        vertical_fov_deg=vertical_fov_deg,
    )
    map_x, map_y, out_w, out_h = build_polar_remap(cx, cy, r_inner, r_outer, out_w, out_h)

    unwrapped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if output_path is None:
        output_path = default_output_path(candidate_path)

    output_dir = os.path.dirname(output_path) or "."
    ensure_directory(output_dir)

    cv2.imwrite(output_path, unwrapped)
    print(f"Saved unwrapped image to {output_path}")

    if show_result:
        cv2.imshow("Unwrapped", unwrapped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Unwrap donut-shaped region into a rectangle.")
    parser.add_argument("json", help="JSON file with center and radii")
    parser.add_argument(
        "--image",
        "-i",
        help="Path to a single image or a directory of images (fallback to image_path in JSON if omitted)",
        default=None,
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Output image path (single image) or directory (batch mode). If omitted for a single image, "
            "defaults to unwrap_<image name> saved next to the source image."
        ),
        default=None,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional output width (pixels). If omitted, computed from mean radius.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional output height (pixels). If omitted, it is computed from the vertical FOV (or the donut thickness when FOV scaling is disabled).",
    )
    parser.add_argument(
        "--vertical-fov",
        type=float,
        default=DEFAULT_VERTICAL_FOV_DEG,
        help=(
            "Camera vertical FOV in degrees used to derive the panorama aspect ratio "
            "when --height is not provided. Set to 0 to keep the legacy pixel-based height."
        ),
    )

    args = parser.parse_args()

    params = load_params(args.json)
    params = (expand_path(params[0]), params[1], params[2], params[3], params[4])

    image_arg = expand_path(args.image)
    output_arg = expand_path(args.output)

    image_target = image_arg if image_arg is not None else params[0]

    if image_target is None:
        parser.error("No image path provided via --image and JSON does not include image_path.")

    if os.path.isdir(image_target):
        if output_arg is None:
            parser.error("When processing a directory, --output must point to an output directory.")
        if os.path.exists(output_arg) and not os.path.isdir(output_arg):
            parser.error("Output path must be a directory when processing a directory of images.")

        ensure_directory(output_arg)
        image_paths = list_images_in_directory(image_target)
        if not image_paths:
            parser.error(f"No supported image files found in directory: {image_target}")

        for img_path in image_paths:
            out_file = os.path.join(output_arg, prefixed_filename(os.path.basename(img_path)))
            unwrap_donut(
                image_path=img_path,
                json_path=args.json,
                output_path=out_file,
                out_w=args.width,
                out_h=args.height,
                vertical_fov_deg=args.vertical_fov,
                show_result=False,
                preloaded_params=params,
            )
        print(f"Processed {len(image_paths)} images from {image_target}")

    elif os.path.isfile(image_target):
        output_path = output_arg
        if output_path is not None and looks_like_directory(output_path):
            ensure_directory(output_path)
            filename = prefixed_filename(os.path.basename(image_target))
            output_path = os.path.join(output_path, filename)

        unwrap_donut(
            image_path=image_target,
            json_path=args.json,
            output_path=output_path,
            out_w=args.width,
            out_h=args.height,
            vertical_fov_deg=args.vertical_fov,
            preloaded_params=params,
        )

    else:
        parser.error(f"Provided image path does not exist: {image_target}")


if __name__ == "__main__":
    main()


'''
Example usage:

python unwrap_donut.py my_image_donut_params.json

Or explicitly:

python unwrap_donut.py my_image_donut_params.json --image my_image.png -o ./outputs/
# -> saves ./outputs/unwrap_my_image.png

python unwrap_donut.py params.json --image ./captures/E151/wp --output ./unwrapped_wp
# -> unwraps every supported image in the folder and saves as unwrap_<original name>



If you want more/less resolution in the unwrapped image:

python unwrap_donut.py my_image_donut_params.json --width 4000 --height 300




python tools/unwrap_donut.py captures/E198/metal/params.json --image captures/E198/metal/donut -o captures/E198/metal/unwrapped
'''
