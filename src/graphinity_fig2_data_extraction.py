"""

Utilities to extract values from curve from a PNG image.

Main function:
- extract_red_curve(image_path, ...) -> dict with pixel coordinates and optional data coords

This is purposely conservative: it returns pixel coordinates for the red solid line labelled
"Experimental_d_dG_645+Reverse_mutation-Non-Binders". Mapping to plot data coordinates requires
either user-supplied transform parameters or additional code to detect axes/ticks.

Example usage is included in the module's __main__ which creates a synthetic image, runs
extraction, and prints a small sample of detected points.

Dependencies: numpy, opencv-python (cv2), scipy (optional), scikit-image (optional), matplotlib (for plotting)
"""

import os
from typing import Optional, Tuple, Dict

import numpy as np
import cv2


def extract_red_curve(
    image_path: str,
    lower_sat: int = 50,
    lower_val: int = 50,
    morph_kernel: int = 3,
    min_contour_points: int = 10,
    return_mask: bool = False,
    pixel_to_data: Optional[Tuple[float, float, float, float]] = None,
) -> Dict:
    """
    Extracts the (x_pixel, y_pixel) coordinates of the dominant red solid line in `image_path`.

    Parameters:
    - image_path: path to the PNG/JPG image
    - lower_sat, lower_val: lower thresholds for saturation and value in HSV to consider "red"
    - morph_kernel: size of morphological kernel used to clean up the mask
    - min_contour_points: minimum number of points required for a found contour to be considered
    - return_mask: if True, include the binary mask used for detection in the returned dict
    - pixel_to_data: optional (scale_x, offset_x, scale_y, offset_y) to transform pixel coords to data coords

    Returns a dict with keys:
    - x_pixels: 1D numpy array of x pixel coordinates (increasing left->right)
    - y_pixels: 1D numpy array of y pixel coordinates (same length as x_pixels), image coordinates (top=0)
    - x_data, y_data: optional transformed coordinates when pixel_to_data is provided
    - mask: optional binary mask used for detection (only if return_mask=True)

    Notes: This function returns pixel coordinates. Mapping to plot units requires a known linear
    transform from pixels to data coordinates (provided via pixel_to_data) or additional axis parsing.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image (cv2 returned None): {image_path}")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red in HSV sits around 0 and 180 -> use two ranges
    lower1 = np.array([0, lower_sat, lower_val])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, lower_sat, lower_val])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological clean-up
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours (line will produce long thin contour)
    contours_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        raise RuntimeError("No red contours found in image with the provided thresholds")

    # Select the contour with the largest number of points (best for long lines)
    best_contour = max(contours, key=lambda c: c.shape[0])

    if best_contour.shape[0] < min_contour_points:
        raise RuntimeError("Found red contour but it is too small to be the expected curve")

    pts = best_contour.reshape(-1, 2)  # (N, 2) as (x, y)

    # For each integer x (0..width-1) compute the median y to get a single-valued function y(x).
    img_h, img_w = mask.shape
    # group points by x pixel (rounded)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, img_w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, img_h - 1)

    uniq_x = np.unique(xs)
    x_pixels_list = []
    y_pixels_list = []
    for x in uniq_x:
        y_vals = ys[xs == x]
        if y_vals.size == 0:
            continue
        y_med = int(np.median(y_vals))
        x_pixels_list.append(x)
        y_pixels_list.append(y_med)

    # Sort by x ascending
    order = np.argsort(x_pixels_list)
    x_pixels = np.array(x_pixels_list)[order]
    y_pixels = np.array(y_pixels_list)[order]

    result = {"x_pixels": x_pixels, "y_pixels": y_pixels}

    if pixel_to_data is not None:
        sx, ox, sy, oy = pixel_to_data
        x_data = x_pixels * sx + ox
        y_data = y_pixels * sy + oy
        result["x_data"] = x_data
        result["y_data"] = y_data

    if return_mask:
        result["mask"] = mask

    return result


def save_curve_csv(out_csv_path: str, x: np.ndarray, y: np.ndarray) -> None:
    """Save x,y arrays to a two-column CSV (header: x,y)."""
    import csv

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])  # header
        for xi, yi in zip(x, y):
            writer.writerow([float(xi), float(yi)])


def compute_pixel_to_data_from_points(
    px_points: Tuple[Tuple[float, float], Tuple[float, float]],
    data_points: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """
    Compute a simple linear transform from pixel coords to data coords assuming separable
    linear transforms for x and y:

      x_data = sx * x_px + ox
      y_data = sy * y_px + oy

    Inputs:
    - px_points: ((x1_px, y1_px), (x2_px, y2_px)) two distinct pixel coordinates on the axes
    - data_points: ((x1_data, y1_data), (x2_data, y2_data)) corresponding data coordinates

    Returns: (sx, ox, sy, oy)

    Note: This requires that the two provided points are not vertically/horizontally degenerate
    for the respective axis mapping. Use points that share the same y for x mapping (or ignore y)
    and same x for y mapping, or simply provide two points and the function will compute sx,ox
    from x components and sy,oy from y components.
    """
    (x1_px, y1_px), (x2_px, y2_px) = px_points
    (x1_data, y1_data), (x2_data, y2_data) = data_points

    if x2_px == x1_px:
        raise ValueError("Pixel x coordinates for the two reference points must be different")
    if y2_px == y1_px:
        # still OK for x mapping; y mapping uses y px values
        pass

    sx = (x2_data - x1_data) / (x2_px - x1_px)
    ox = x1_data - sx * x1_px

    if y2_px == y1_px:
        # fallback: use y data from input points for sy (if their data y differ use that)
        if y2_data == y1_data:
            raise ValueError("Pixel y coordinates for the two reference points must be different to compute y mapping")

    if y2_px == y1_px:
        # If pixel y same (unlikely), compute sy from data y differences using px differences in x
        sy = (y2_data - y1_data) / (y2_px - y1_px + 1e-12)
    else:
        sy = (y2_data - y1_data) / (y2_px - y1_px)
    oy = y1_data - sy * y1_px

    return sx, ox, sy, oy


if __name__ == "__main__":
    # Quick smoke test: create a synthetic image with a red polyline and run extraction.
    import tempfile

    w, h = 400, 200
    canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)

    # Draw a noisy sine-like red curve
    pts = []
    for x in range(10, w - 10, 2):
        y = int((h / 2) + 40 * np.sin((x / w) * 6.0) + np.random.randn() * 1.0)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (0, 0, 255), 2)  # BGR red

    tmp = os.path.join(tempfile.gettempdir(), "test_red_curve.png")
    cv2.imwrite(tmp, canvas)
    print("Wrote test image to:", tmp)

    res = extract_red_curve(tmp, lower_sat=80, lower_val=80, morph_kernel=3, return_mask=False)
    x_pixels = res["x_pixels"]
    y_pixels = res["y_pixels"]
    print("Detected points (sample 10):")
    for xi, yi in list(zip(x_pixels, y_pixels))[:10]:
        print(xi, yi)

    # Save to CSV next to the temp image
    csv_out = tmp.replace('.png', '.csv')
    save_curve_csv(csv_out, x_pixels, y_pixels)
    print("Saved detected curve to:", csv_out)
