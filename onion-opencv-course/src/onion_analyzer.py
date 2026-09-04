"""
onion_analyzer.py
=================
A rule-based Onion Quality Image Analyzer (traditional computer vision).

Pipeline:
    read -> resize -> grayscale -> blur -> Otsu threshold -> morphology
          -> largest contour -> mask -> features -> rules -> result

Every intermediate image is saved so it can be used in the book as a
REAL screenshot (the output of actually running this code).

Usage:
    python onion_analyzer.py <image_path> [output_dir]

Demo thresholds are intentionally simple and are meant to be tuned
against a real, labeled dataset (see the book, Chapter 20 & 24).
"""
import sys
import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------
# STEP 1: READ + RESIZE
# ----------------------------------------------------------------------------
def read_and_resize(path, width=640):
    img = cv2.imread(path)                     # BGR
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if w > width:
        scale = width / w
        img = cv2.resize(img, (width, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img

# ----------------------------------------------------------------------------
# STEP 2: SEGMENT THE ONION (background removal)
# ----------------------------------------------------------------------------
def segment_onion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu thresholding finds a cutoff that best separates the
    # darker onion from the brighter background.
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations clean small holes / specks.
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep the largest contour = the onion.
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour found — check the image/background.")
    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)   # filled

    masked_img = cv2.bitwise_and(img, img, mask=mask)  # onion only
    return gray, blurred, thresh, opened, largest, mask, masked_img

# ----------------------------------------------------------------------------
# STEP 3: FEATURES
# ----------------------------------------------------------------------------
def extract_features(mask, gray, hsv):
    # --- shape ---
    area = float(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(c, True))
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(h) / float(w) if w > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0

    # --- colour (inside the onion only) ---
    mean_bgr = cv2.mean(hsv, mask=mask)          # note: mean over HSV here
    mean_b = mean_bgr[0]; mean_g = mean_bgr[1]; mean_r = mean_bgr[2]
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    std_gray = float(gray_masked[np.nonzero(mask)].std())

    # --- visible defect proxies (DEMO thresholds) ---
    dark_ratio = float(np.mean(gray[mask > 0] < 70))           # very dark px
    h, s, v = cv2.split(hsv)
    brown = ((h >= 8) & (h <= 25) & (s >= 40) & (v <= 200)).astype(np.uint8)
    brown_ratio = float(brown[mask > 0].mean())                # brown/rot px
    defect_ratio = max(dark_ratio, brown_ratio)

    return {
        "area": area, "perimeter": perimeter, "circularity": circularity,
        "aspect_ratio": aspect_ratio, "extent": extent,
        "mean_hsv": (round(mean_b, 1), round(mean_g, 1), round(mean_r, 1)),
        "std_gray": round(std_gray, 1),
        "dark_ratio": round(dark_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "defect_ratio": round(defect_ratio, 4),
    }

# ----------------------------------------------------------------------------
# STEP 4: RULES (DEMO — tune against a labeled dataset)
# ----------------------------------------------------------------------------
def classify(feats):
    dr, br = feats["dark_ratio"], feats["brown_ratio"]
    if dr >= 0.12 or br >= 0.40:
        return "ROTTEN", (0, 0, 255)
    if dr >= 0.04 or br >= 0.15:
        return "DAMAGED", (0, 165, 255)
    return "GOOD", (0, 200, 0)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def analyze(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = read_and_resize(path)

    gray, blurred, thresh, opened, largest, mask, masked_img = segment_onion(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = extract_features(mask, gray, hsv)
    label, color = classify(feats)

    # --- annotated result ---
    result = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 100, 0), 3)          # outline
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)      # box
    cv2.putText(result, f"{label}  (defect={feats['defect_ratio']})",
                (x, max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)

    # --- save every stage ---
    def save(name, arr):
        cv2.imwrite(os.path.join(out_dir, name), arr)

    save("01_original.jpg", img)
    save("02_grayscale.jpg", gray)
    save("03_blurred.jpg", blurred)
    save("04_threshold.jpg", thresh)
    save("05_morphology.jpg", opened)
    contour_img = img.copy()
    cv2.drawContours(contour_img, [largest], -1, (0, 255, 0), 3)
    save("06_contour.jpg", contour_img)
    save("07_mask.jpg", mask)
    save("08_masked_onion.jpg", masked_img)
    save("09_result.jpg", result)

    # hsv visual
    save("10_hsv.jpg", hsv)

    return label, feats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onion_analyzer.py <image_path> [output_dir]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    label, feats = analyze(src, out)
    print(f"RESULT: {label}")
    for k, v in feats.items():
        print(f"  {k:12s} = {v}")
