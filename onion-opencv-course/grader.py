"""
grader.py  (SIH26031 — Step 2)
================================
The CORE engine of the onion grading app.

Given ONE photo of a batch of onions it:
  1. DETECTS each onion          -> threshold + contours + WATERSHED (splits touching onions)
  2. COUNTS them EXACTLY          -> one result per onion
  3. CLASSIFIES each              -> GOOD / DAMAGED / ROTTEN / SPROUTED / UNDERSIZED
  4. MEASURES diameter in mm      -> using a reference coin in the photo
  5. GRADES                       -> Grade A (45-65mm) / URS (35-70mm) / REJECT
  6. REPORTS                      -> annotated image + JSON + text + report card

Run it:
    python grader.py images/demo_batch.jpg 27
    python grader.py images/test_touching.jpg 27

Reference coin sizes: Rs.10 / Rs.2 = 27 mm, Rs.5 = 23 mm, Rs.1 = 22 mm.
If no coin is found it assumes the median onion is 55 mm and WARNS you.

HONESTY: this grades VISIBLE SURFACE quality only. A photo cannot see
inside an onion (internal rot, moisture) — we state this in every report.
"""
import os
import sys
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------ CONFIG -------------------------------------
REF_DIAMETER_MM = 27.0          # default reference coin (Rs.10 / Rs.2)
MIN_AREA        = 1500          # ignore specks smaller than this (px^2)
GRADE_A_MM      = (45, 65)      # strict procurement spec
GRADE_URS_MM    = (35, 70)      # relaxed spec ("Grade URS")
UNDERSIZE_MM    = 35.0          # below this = UNDERSIZED
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

COLORS = {
    "GOOD": (0, 180, 0), "DAMAGED": (0, 165, 255), "ROTTEN": (0, 0, 255),
    "SPROUTED": (0, 200, 200), "UNDERSIZED": (255, 140, 0), "COIN": (80, 80, 80),
}

# ----------------------------------------------------------------------------
# 1. READ IMAGE
# ----------------------------------------------------------------------------
def read_image(path, max_w=1200):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if w > max_w:                       # keep the image a manageable size
        img = cv2.resize(img, (max_w, int(h * max_w / w)),
                         interpolation=cv2.INTER_AREA)
    return img

# ----------------------------------------------------------------------------
# 2. DETECT (find every onion)
# ----------------------------------------------------------------------------
def find_objects(img):
    """Turn the photo into a binary image where onions = white blobs,
    then return one contour per blob."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)             # remove noise
    # Otsu picks the best cutoff: onion (dark) vs background (light)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)   # fill holes
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)    # drop specks
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    return th, contours

def _watershed_blob(img, c):
    """Split ONE merged blob into its separate bulbs using watershed.

    The distance-transform 'peaks' locate each bulb's centre. A single fixed
    threshold fails (shallow vs deep valleys), so we SWEEP thresholds from
    high to low and use the first one that finds exactly 2 bulbs.
    Returns the sub-contours (or the original blob if it can't be split)."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)

    k3 = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, k3, iterations=3)                 # surely background
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)           # distance to edges

    best = None
    for t in (0.60, 0.55, 0.50, 0.45, 0.40, 0.35):               # sweep thresholds
        _, sf = cv2.threshold(dist, t * dist.max(), 255, 0)
        sf = np.uint8(sf)
        n, lab = cv2.connectedComponents(sf)
        if n - 1 == 2:                                           # exactly 2 bulbs
            best = lab
            break
    if best is None:
        return [c]                                               # can't split

    unknown = cv2.subtract(sure_bg, np.uint8(best > 0) * 255)
    markers = best.astype(np.int32) + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)                        # draw the cut line

    parts = []
    for lbl in range(2, int(markers.max()) + 1):
        m = np.uint8(markers == lbl) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            pc = max(cs, key=cv2.contourArea)
            if cv2.contourArea(pc) >= MIN_AREA:
                parts.append(pc)
    return parts if len(parts) > 1 else [c]

def split_merged(img, contours):
    """A blob that is BIGGER than a typical onion AND not round is probably
    2+ onions touching -> split it with watershed."""
    if not contours:
        return contours
    areas = np.array([cv2.contourArea(c) for c in contours])
    med = float(np.median(areas))
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        circ = (4 * np.pi * area) / (perim ** 2) if perim else 0.0
        # merged clue: clearly bigger than the median AND squashed/not-round
        if area > 1.3 * med and circ < 0.75:
            out.extend(_watershed_blob(img, c))
        else:
            out.append(c)
    return out

# ----------------------------------------------------------------------------
# 3. GEOMETRY of each onion
# ----------------------------------------------------------------------------
def geom(c):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
    x, y, w, h = cv2.boundingRect(c)
    d_px = 2 * np.sqrt(area / np.pi)       # equivalent diameter in PIXELS
    return dict(area=area, perim=perim, circ=circ, box=(x, y, w, h), d_px=d_px)

def detect_coin(objs):
    """The reference coin = a ROUND object clearly SMALLER than the onions."""
    if not objs:
        return None
    areas = np.array([o["area"] for o in objs])
    med = float(np.median(areas))
    cands = [o for o in objs if o["circ"] >= 0.80 and o["area"] < 0.6 * med]
    return max(cands, key=lambda o: o["circ"]) if cands else None

# ----------------------------------------------------------------------------
# 4. CLASSIFY + GRADE
# ----------------------------------------------------------------------------
def classify(green_ratio, dark_ratio, brown_ratio, d_mm):
    """Rules over visible features (tune these on a labeled dataset later)."""
    if green_ratio >= 0.02:                              # green = sprout
        return "SPROUTED"
    if dark_ratio >= 0.12 or brown_ratio >= 0.40:        # lots of dark/brown = rot
        return "ROTTEN"
    if dark_ratio >= 0.04 or brown_ratio >= 0.15:        # some blemish = damaged
        return "DAMAGED"
    if d_mm < UNDERSIZE_MM:
        return "UNDERSIZED"
    return "GOOD"

def grade_for(label, d_mm):
    if label != "GOOD":
        return "REJECT"
    if GRADE_A_MM[0] <= d_mm <= GRADE_A_MM[1]:
        return "Grade A"
    if GRADE_URS_MM[0] <= d_mm <= GRADE_URS_MM[1]:
        return "Grade URS"
    return "UNGRADED"

# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def analyze_batch(path, ref_mm=REF_DIAMETER_MM, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    img = read_image(path)

    _, contours = find_objects(img)
    if not contours:
        raise RuntimeError("No objects found — check lighting/background.")
    contours = split_merged(img, contours)               # split touching onions
    objs = [geom(c) for c in contours]
    coin = detect_coin(objs)

    # ---- scale: pixels -> millimetres ----
    if coin:
        px_per_mm = coin["d_px"] / ref_mm
        scale_note = f"scale: reference = {ref_mm:.0f} mm"
    else:
        median_d = float(np.median([o["d_px"] for o in objs]))
        px_per_mm = median_d / 55.0
        scale_note = "WARNING: no reference object detected — assumed median onion = 55 mm"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # colour masks (healthy golden skin is BRIGHT; defects are DARKER -> V<=160)
    green = ((H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)).astype(np.uint8)
    brown = ((H >= 8) & (H <= 25) & (S >= 60) & (V <= 160)).astype(np.uint8)

    coin_idx = objs.index(coin) if coin else None
    results, annotated = [], img.copy()
    for i, (o, c) in enumerate(zip(objs, contours)):
        if coin_idx is not None and i == coin_idx:
            continue                                  # the coin is the ruler, not an onion
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        px = mask > 0
        d_mm = o["d_px"] / px_per_mm
        green_ratio = float(green[px].mean())
        brown_ratio = float(brown[px].mean())
        dark_ratio  = float(np.mean(gray[px] < 70))
        label = classify(green_ratio, dark_ratio, brown_ratio, d_mm)
        grade = grade_for(label, d_mm)
        results.append(dict(d_mm=round(d_mm, 1), label=label, grade=grade,
                            circ=round(o["circ"], 2),
                            brown=round(brown_ratio, 3),
                            dark=round(dark_ratio, 3),
                            green=round(green_ratio, 3)))
        # ---- draw ----
        x, y, w, h = o["box"]
        col = COLORS[label]
        cv2.drawContours(annotated, [c], -1, col, 3)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(annotated, f"{label} {d_mm:.0f}mm [{grade}]",
                    (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if coin:
        x, y, w, h = coin["box"]
        cv2.putText(annotated, "REF (scale)", (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["COIN"], 2)

    # ---- percentages ----
    total = len(results)
    n_a = sum(1 for r in results if r["grade"] == "Grade A")
    n_u = sum(1 for r in results if r["grade"] == "Grade URS")
    summary = dict(total=total,
                   grade_a=round(100 * n_a / total, 1) if total else 0,
                   urs=round(100 * n_u / total, 1) if total else 0,
                   reject=round(100 * (total - n_a - n_u) / total, 1) if total else 0,
                   counts=dict(GradeA=n_a, URS=n_u, Reject=total - n_a - n_u),
                   scale=scale_note)
    report = dict(summary=summary, onions=results, ref_diameter_mm=ref_mm)

    # ---- save outputs ----
    cv2.imwrite(os.path.join(out_dir, "annotated.jpg"), annotated)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    txt = [f"ONION QUALITY REPORT", f"{'='*22}", scale_note,
           f"Total onions: {total}",
           f"Grade A: {summary['grade_a']}%   URS: {summary['urs']}%   "
           f"Reject: {summary['reject']}%", ""]
    for i, r in enumerate(results, 1):
        txt.append(f"{i}. {r['label']:10s} {r['d_mm']:5.1f} mm   {r['grade']}")
    txt.append("\nNOTE: visible surface analysis only — cannot detect internal defects.")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(txt))
    make_report_card(report, os.path.join(out_dir, "report_card.jpg"))

    return report, annotated

def make_report_card(report, path):
    W, Hh = 760, 150 + 46 * len(report["onions"]) + 60
    img = Image.new("RGB", (W, Hh), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT_BOLD, 30)
    f_h = ImageFont.truetype(FONT_BOLD, 20)
    f_t = ImageFont.truetype(FONT_MONO, 16)
    d.rectangle([0, 0, W, 74], fill=(122, 59, 18))
    d.text((24, 16), "ONION QUALITY REPORT  (SIH26031)", font=f_title, fill=(255, 244, 230))
    d.text((24, 88), report["summary"]["scale"], font=f_t, fill=(90, 90, 90))
    y = 116
    s = report["summary"]
    d.text((24, y), f"Total onions: {s['total']}    Grade A: {s['grade_a']}%    "
                    f"URS: {s['urs']}%    Reject: {s['reject']}%",
           font=f_h, fill=(40, 40, 40))
    y += 40
    d.text((24, y), " #   RESULT        DIAMETER   GRADE", font=f_t, fill=(120, 120, 120))
    y += 24
    for i, r in enumerate(report["onions"], 1):
        d.text((24, y), f"{i:2d}   {r['label']:<11s} {r['d_mm']:>6.1f} mm   {r['grade']}",
               font=f_t, fill=(30, 30, 30))
        y += 26
    d.text((24, y + 6),
           "Visible-surface analysis only. Internal defects cannot be detected from a photo.",
           font=f_t, fill=(160, 60, 60))
    img.save(path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grader.py <image_path> [reference_mm]")
        sys.exit(1)
    ref = float(sys.argv[2]) if len(sys.argv) > 2 else REF_DIAMETER_MM
    rep, _ = analyze_batch(sys.argv[1], ref)
    print(json.dumps(rep["summary"], indent=2))
    for r in rep["onions"]:
        print(f"  {r['label']:10s} {r['d_mm']:5.1f} mm   {r['grade']}")
