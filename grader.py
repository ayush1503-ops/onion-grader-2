#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 grader.py - ONION QUALITY GRADER  (the "brain" of the app)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT THIS FILE DOES (one line):
    Takes ONE photo of a batch of onions -> counts the onions,
    measures each one in millimetres, checks the VISIBLE surface of
    each one, grades the batch, and writes 4 report files.

THE 4 REPORT FILES (saved in outputs/, one set per photo):
    <photo>_annotated.jpg    photo with every onion boxed+labeled+sized
    <photo>_report.json      machine-readable report (for apps)
    <photo>_report.txt       human-readable report
    <photo>_report_card.jpg  a nice "report card" picture of the batch

THE PIPELINE (classic computer vision - no training needed):
    photo
      1. resize (max width 1200 px)
      2. grayscale
      3. Gaussian blur 7x7          -> calm down noise
      4. Otsu threshold             -> separate onions from background
      5. morphology close + open    -> tidy the mask, remove specks
      6. findContours               -> candidate blobs
      7. WATERSHED (cv2.watershed)  -> split onions that TOUCH
      8. coin detection             -> convert pixels -> millimetres
      9. HSV + darkness features    -> measured INSIDE each onion only
     10. rule-based classification  -> GOOD/DAMAGED/ROTTEN/SPROUTED/UNDERSIZED
     11. grading                    -> Grade A / URS / REJECT + percentages
     12. reports                    -> the 4 files above

HONESTY RULES (never remove this):
    * This tool grades VISIBLE SURFACE quality only.
      A normal photo CANNOT detect internal rot, internal damage,
      or internal moisture.
    * All thresholds below are DEMO STARTING POINTS. They are NOT
      verified accuracy. Real accuracy must be measured on a labeled
      test set (photos checked by a human expert).

HOW TO RUN:
    python grader.py <photo.jpg> --coin-mm 27
    (Rs.10 / Rs.2 coin = 27 mm, Rs.5 = 23 mm, Rs.1 = 22 mm)
"""

import argparse
import base64
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------
# SETTINGS - all the "knobs" in one place
# ------------------------------------------------------------------
MAX_WIDTH         = 1200   # images wider than this are resized down
MIN_AREA_PX       = 1500   # ignore blobs smaller than this (specks/noise)
MIN_ONION_FRAC    = 0.20   # SECONDARY-pass candidates must also be at least
                           # this fraction of the median onion area. The old
                           # absolute 1500 px floor let TEXTURED backgrounds
                           # (jute bag, straw, cloth) through - each woven
                           # strand fragment is > 1500 px but far smaller
                           # than an onion, and they were counted as extra
                           # onions. A real missed onion is roughly onion-
                           # sized, so 20% of the median is a safe floor.
BLUR_K            = (7, 7) # Gaussian blur kernel size
MORPH_K           = (7, 7) # morphology kernel size
MERGED_FACTOR     = 1.3    # blob area > 1.3 x median  -> maybe 2 touching onions
CIRC_SPLIT_MAX    = 0.75   # ...and circularity < 0.75 (not round) -> try split
COIN_CIRC_MIN     = 0.80   # a coin must be very round
COIN_AREA_MAX     = 0.60   # ...and much smaller than a typical onion
COIN_SAT_MAX      = 90     # ...and metallic: mean saturation this low
                           # (onion skin sits around 100-140, metal ~0)
# Seed-level sweep for watershed. The original spec range (0.60 -> 0.35)
# gives strong, fat seeds and is tried FIRST. If no level gives exactly
# 2 seed islands (e.g. morphology rounded the "waist" between two
# touching onions), we climb to higher levels as a rescue.
WS_SWEEP_MAIN  = (0.60, 0.55, 0.50, 0.45, 0.40, 0.35)
WS_SWEEP_EXTRA = (0.85, 0.80, 0.75, 0.70, 0.65)
# low tail: seeds for MOSTLY-HIDDEN onions (thin visible crescents)
WS_SWEEP_LOW   = (0.30, 0.25, 0.20)
MIN_PIECE_FRAC    = 0.30   # a valid split: each piece >= 30% of the blob
SPLIT_OVERLAP_MAX = 0.25   # ...and NO piece may sit mostly INSIDE another
                           # piece's outline. When watershed mis-splits ONE
                           # onion, one seed's flood wraps around the other
                           # -> a nested "ring" piece around the other piece.
                           # A ring + its core is one onion counted twice,
                           # so such splits are rejected (see _pieces_overlap).
FALLBACK_ONION_MM = 55.0   # used ONLY when no coin is found (a GUESS!)

# Pile-layer estimate (which "layer" of the pile is each onion in?).
# visibility = EXTENT = region area / its convex-hull area.
# A fully visible onion is a full disc -> extent ~1.0. An onion covered
# by others shows only a crescent -> extent drops (hull stays big).
# NOTE: bands are DEMO starting points - calibrate on real pile photos.
LAYER_BANDS = [(0.80, "L1", "Top layer (fully visible)"),
               (0.45, "L2", "Middle layer (partly covered)"),
               (0.00, "L3", "Buried layer (mostly hidden)")]
LAYER_NOTE = ("Layer assignment is an occlusion ESTIMATE from one 2D photo. "
              "Onions FULLY hidden under others cannot be seen or counted "
              "by any camera - mix/turn the pile for a complete check.")

# Weight estimation: onion mass grows with the CUBE of its diameter.
# mass_g = WEIGHT_K * diameter_mm^3. Default 0.00051 gives ~85 g for a
# 55 mm onion. CALIBRATE ONCE: weigh 10 onions on a kitchen scale,
# then  weight_k = total_g / (0.00051 * 10 * mean_diameter_mm^3).
WEIGHT_K_G_PER_MM3 = 0.00051
BORDER_PX = 12             # border ring used to sense light/dark background

# Indian coin diameters (approximate, in mm)
COIN_MENU = {"10": 27.0, "2": 27.0, "5": 23.0, "1": 22.0}

# Grading sizes in mm
GRADE_A_MM    = (45.0, 65.0)   # Grade A window
URS_MM        = (35.0, 70.0)   # URS = relaxed specification window
UNDERSIZED_MM = 35.0           # below this diameter -> "UNDERSIZED"

# --- classification thresholds (DEMO values - MUST be tuned on real data!) ---
GREEN_SPROUTED = 0.02  # >= 2%  green pixels       -> SPROUTED
DARK_ROTTEN    = 0.12  # >= 12% very dark pixels   -> ROTTEN
BROWN_ROTTEN   = 0.40  # >= 40% dark-brown pixels  -> ROTTEN
DARK_DAMAGED   = 0.04  # >= 4%  very dark pixels   -> DAMAGED
BROWN_DAMAGED  = 0.15  # >= 15% dark-brown pixels  -> DAMAGED

DISCLAIMER = ("VISIBLE SURFACE ANALYSIS ONLY - a normal photo cannot detect "
              "internal rot, internal damage or internal moisture. "
              "Thresholds are demo starting points; real accuracy must be "
              "measured on a labeled test set.")

# BGR drawing colors, one per class
CLASS_COLORS = {
    "GOOD":       (80, 200, 80),
    "DAMAGED":    (0, 165, 255),
    "ROTTEN":     (0, 0, 255),
    "SPROUTED":   (255, 0, 255),
    "UNDERSIZED": (255, 200, 0),
}


# ------------------------------------------------------------------
# STEP 1 - read the photo (and shrink it if it is huge)
# ------------------------------------------------------------------
def read_image(path):
    """Load a photo with OpenCV. Big photos are shrunk to width 1200."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        new_h = int(h * MAX_WIDTH / w)
        img = cv2.resize(img, (MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return img


def _fit_width(bgr):
    """Same size cap as read_image, but for an already-loaded frame."""
    h, w = bgr.shape[:2]
    if w > MAX_WIDTH:
        bgr = cv2.resize(bgr, (MAX_WIDTH, int(h * MAX_WIDTH / w)),
                         interpolation=cv2.INTER_AREA)
    return bgr


def circularity(cnt):
    """How round a shape is: 1.0 = perfect circle, lower = less round."""
    per = cv2.arcLength(cnt, True)
    if per == 0:
        return 0.0
    return 4.0 * math.pi * cv2.contourArea(cnt) / (per * per)


# ------------------------------------------------------------------
# STEPS 2-6 - build the "object mask" and find candidate blobs
# ------------------------------------------------------------------
def flatten_illumination(gray):
    """
    Remove UNEVEN LIGHTING: divide the image by its own heavily blurred
    version. Slow brightness changes (shadows, one dark corner) vanish;
    the onions keep their LOCAL contrast. Classic photometric trick.
    """
    bg = cv2.GaussianBlur(gray, (0, 0), 51)
    return cv2.divide(gray, bg, scale=255)


def fill_holes(mask):
    """Fill interior holes of a mask (flood fill from the corner)."""
    h, w = mask.shape
    ff = mask.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def refine_local_otsu(gray, contour, pad=15, hsv=None):
    """
    Tighten a rough candidate region with a LOCAL Otsu threshold on its
    crop (auto light/dark polarity per crop). Returns a cleaner contour
    in full-image coordinates, or the original candidate.

    When colour is available the SATURATION channel is tried FIRST:
    shadows shift brightness but not colourfulness, so on a crop that
    straddles a light/shadow seam (where onion and background can share
    the same brightness) saturation still cuts cleanly. Brightness is
    the fallback for grey-on-grey cases.
    """
    x, y, w, h = cv2.boundingRect(contour)
    H, W = gray.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    cx, cy = x + w // 2 - x0, y + h // 2 - y0

    def pick(channel):
        crop = channel[y0:y1, x0:x1]
        cnts = get_blobs(mask_from_gray(crop))
        best, best_a = None, 0.0
        for c in cnts:
            if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0:
                a = cv2.contourArea(c)
                if a > best_a:
                    best, best_a = c, a
        if best is None or best_a < 0.25 * w * h or best_a > 1.6 * w * h:
            return None                 # local cut did not agree
        return best + np.array([x0, y0])

    if hsv is not None:
        sat = pick(hsv[:, :, 1])
        if sat is not None:
            return sat
    lum = pick(gray)
    if lum is not None:
        return lum
    return contour


def local_contrast_mask(gray, hsv=None):
    """
    Mask of everything that differs from its LOCAL surroundings.
    |gray - blurred| > 15 catches darker AND brighter onions (the plain
    divide-by-blur trick clips bright objects into the background).
    Only the onion RIM survives the first cut (interiors match their own
    blur), so we close gaps and flood-fill the interiors into solid disks.

    A SECOND, illumination-proof cue is unioned in when colour is given:
    |saturation - blurred saturation|. A shadow changes brightness but
    not colourfulness, so this ring stays CLOSED even where a shadow is
    exactly as bright as the onion rim - the case that opens gaps in the
    brightness ring and chops shadow-side onions into fragments.
    """
    bg = cv2.GaussianBlur(gray, (0, 0), 51)
    diff = cv2.absdiff(gray, bg)
    _, m = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    if hsv is not None:
        sat = hsv[:, :, 1]
        sbg = cv2.GaussianBlur(sat, (0, 0), 51)
        sdiff = cv2.absdiff(sat, sbg)
        _, ms = cv2.threshold(sdiff, 25, 255, cv2.THRESH_BINARY)
        m = cv2.bitwise_or(m, ms)
    k = np.ones(MORPH_K, np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    return fill_holes(m)


def mask_from_gray(gray):
    """Threshold ONE gray image into an object mask (auto polarity)."""
    blur = cv2.GaussianBlur(gray, BLUR_K, 0)                  # step 3
    # Otsu computes the best black/white cut automatically.
    otsu_t, _ = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # sample the border ring (skip 1 px to stay inside the image)
    h, w = gray.shape
    b = max(2, min(BORDER_PX, h // 8, w // 8))
    border = np.ones((h, w), bool)
    border[b:h - b, b:w - b] = False
    border_mean = float(gray[border].mean()) if border.any() else 255.0
    if border_mean >= otsu_t:
        # light background -> onions are DARKER -> keep the dark side
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # dark background -> onions are BRIGHTER -> keep the bright side
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones(MORPH_K, np.uint8)
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return mask


def make_object_mask(bgr):
    """Return (gray, mask). mask is white where we think OBJECTS are.

    Works on ANY background: we sample the BORDER of the photo (the
    edge is almost always background). If the border is LIGHT, onions
    must be the dark side of Otsu's cut; if the border is DARK, onions
    are the bright side. This is how it handles white trays AND dark
    gunny bags / trays without any settings."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)              # step 2
    return gray, mask_from_gray(gray)


def get_blobs(mask):
    """Find outlines (contours) of all blobs big enough to be onions."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)           # step 6
    return [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]


# ------------------------------------------------------------------
# STEP 7 - WATERSHED: split touching onions (2 or MORE per blob)
# ------------------------------------------------------------------
def watershed_split(bgr, contour, expected=2):
    """
    Try to split a merged blob of TOUCHING onions using cv2.watershed.

    Idea in simple words:
      - The distance transform gives every pixel a score: how far it is
        from the edge of the blob. The onion CENTRES have the highest
        scores.
      - We try several cut-off levels (the "sweep"). At the right level
        exactly `expected` islands remain = one seed per onion.
      - cv2.watershed then "floods" the blob from those seeds and the
        flood lines are where we cut.

    `expected` = how many onions we think are in this blob
    (estimated from blob area / median single-onion area).

    Returns a list of `expected` contours on success, or [contour].
    """
    m = np.zeros(bgr.shape[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)

    # sure background = blob grown a bit outward
    sure_bg = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3)

    # distance transform: how deep inside the blob each pixel is
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)

    sure_fg, seed_labels = None, None
    for t in WS_SWEEP_MAIN + WS_SWEEP_EXTRA + WS_SWEEP_LOW:
        # normal sweep first, deep-waist rescue, then thin-crescent tail
        _, cand = cv2.threshold(dist, t * float(dist.max()), 255,
                                cv2.THRESH_BINARY)
        cand = np.uint8(cand)
        n, labels = cv2.connectedComponents(cand)
        if n - 1 == expected:     # exactly `expected` islands = expected seeds
            sure_fg, seed_labels = cand, labels
            break
    if sure_fg is None:
        return [contour]          # could not split -> keep as one blob

    unknown = cv2.subtract(sure_bg, sure_fg)      # area watershed must decide
    markers = (seed_labels + 1).astype(np.int32)  # background=1, seeds=2..n+1
    markers[unknown == 255] = 0                   # 0 = "you decide"
    cv2.watershed(bgr, markers)

    out = []
    for lab in range(2, expected + 2):            # the flooded onions
        piece = np.uint8(markers == lab) * 255
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        out += [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]

    # sanity check: a REAL split gives chunks of roughly onion size.
    # If one "piece" is a tiny sliver, the cut was bad -> keep 1 blob.
    if len(out) == expected:
        merged_area = cv2.contourArea(contour)
        smallest = min(cv2.contourArea(c) for c in out)
        if smallest >= (0.40 / expected) * merged_area:
            return out
    return [contour]


def hough_split(bgr, mask, contour, median_area, r_hint=None):
    """
    Rescue split for OCCLUDED piles (one onion lying ON another).

    Why needed: the distance transform has only ONE basin when an onion
    covers another deeply (no "waist" between the centres), so no
    threshold can separate the seeds. But the hidden onion's rim still
    shows as a circular arc in the photo - so:

      1. cv2.HoughCircles searches the blob region for onion-sized
         circles (minDist is small - heavily overlapped onions have
         CLOSE centres).
      2. Each found circle whose centre lies inside the blob becomes a
         seed (nearest-centre zones).
      3. cv2.watershed refines the cut along the REAL edges (the rim
         between the two onions), so each onion keeps its true surface.

    r_hint: optional onion radius in px. Pass it when median_area is
    meaningless for sizing - the HEAP case (one merged blob): the
    median of ONE blob is the blob itself, so the derived radius would
    be huge and every circle would be rejected.

    Returns (regions, vis_hints): contours plus a visibility guess
    per region (None = measure with the convex-hull extent later).
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 8
    H, W = mask.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    roi = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (7, 7), 0)

    r_med = float(r_hint) if r_hint else math.sqrt(median_area / math.pi)
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=0.6 * r_med, param1=110, param2=25,
                               minRadius=int(0.55 * r_med),
                               maxRadius=int(1.45 * r_med))
    if circles is None:
        return [], []

    blob = mask[y0:y1, x0:x1] > 0
    accepted = []
    for cx, cy, r in circles[0]:
        gx, gy = int(round(cx)), int(round(cy))
        if not (0 <= gy < blob.shape[0] and 0 <= gx < blob.shape[1]):
            continue
        if not blob[gy, gx]:
            continue
        # drop only near-identical circles (occluded centres are CLOSE!)
        if all((cx - ax) ** 2 + (cy - ay) ** 2 > (0.45 * r_med) ** 2
               for ax, ay, _r in accepted):
            accepted.append((cx, cy, r))
    if not accepted:
        return [], []

    h_roi, w_roi = blob.shape
    # SMALL circular cores as watershed seeds. Do NOT seed the whole
    # nearest-centre zone - watershed must flood across the REAL image
    # edges (the rim between the onions) to find the true cut.
    seed_img = np.zeros((h_roi, w_roi), np.int32)
    for k, (cx, cy, r) in enumerate(accepted):
        core = np.zeros((h_roi, w_roi), np.uint8)
        cv2.circle(core, (int(round(cx)), int(round(cy))),
                   max(4, int(0.35 * r)), 255, -1)
        seed_img[(core > 0) & blob] = k + 2
    seed_labels_present = set(np.unique(seed_img)) - {0}
    if len(seed_labels_present) < 1:
        return [], []

    # markers: background ring (outside blob) = 1, seeds = 2..k+2,
    # everything else inside the blob = 0 ("watershed decides")
    blob_u8 = blob.astype(np.uint8)
    outside = cv2.dilate(blob_u8 * 255, np.ones((3, 3), np.uint8),
                         iterations=2)
    markers = np.zeros((h_roi, w_roi), np.int32)
    markers[(outside > 0) & (~blob)] = 1
    markers[seed_img > 0] = seed_img[seed_img > 0]
    cv2.watershed(bgr[y0:y1, x0:x1], markers)

    # minimum piece size: relative to the REFERENCE onion size. In heap
    # mode (r_hint given) median_area is the whole merged blob, so we
    # size the threshold from the hinted onion circle instead.
    ref_area = (math.pi * r_med * r_med) if r_hint else median_area
    min_area = max(MIN_AREA_PX, 0.15 * ref_area)
    regions, vis_hints = [], []
    # the flooded onion regions. For a circle-piece the best occlusion
    # measure is: visible area / true circle area (the Hough circle IS
    # the onion's true size, so this ratio survives watershed leaks).
    for k, (cx, cy, r) in enumerate(accepted):
        piece = np.uint8(markers == k + 2) * 255
        piece = cv2.bitwise_and(piece, blob_u8 * 255)
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c2 in cnts:
            a = cv2.contourArea(c2)
            if a >= min_area:
                regions.append(c2 + np.array([x0, y0]))
                vis_hints.append(float(min(1.0, a / float(math.pi * r * r))))

    # the area the BACKGROUND flood claimed inside the blob = exposed
    # surface of an onion Hough never found a circle for (no hint ->
    # the caller falls back to the convex-hull extent)
    leftover = (markers == 1) & blob
    if int(leftover.sum()) >= 0.35 * median_area:
        piece = leftover.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c2 in cnts:
            if cv2.contourArea(c2) >= min_area:
                regions.append(c2 + np.array([x0, y0]))
                vis_hints.append(None)

    if len(regions) < 2:
        return [], []
    return regions, vis_hints


def _bbox_iou(a, b):
    """IoU of two bboxes (x, y, w, h) - used to de-duplicate detections."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _pieces_overlap(pieces, max_frac=SPLIT_OVERLAP_MAX):
    """True when "split" pieces are NESTED instead of lying side by side.

    Why: cv2.watershed can mis-split ONE elongated/oval onion into two
    when one seed's flood wraps around the other - the losing flood
    survives as a RING-shaped piece around the winner (or the winner as
    an island inside the ring). That is the SAME onion emitted twice.

    Real touching onions always sit SIDE BY SIDE, so almost none of one
    piece's pixels fall inside the other piece's filled outline (only a
    ragged watershed line interleaves). A piece that is mostly INSIDE
    another's filled outline is therefore a nested ring/core pair.

    Measure (every ordered pair i, j): the fraction of piece i's pixels
    that lie inside piece j's FILLED outline. Note the fill matters:
    a ring piece's outline is the whole blob outline, so the core in its
    "hole" counts as inside it. Any fraction > max_frac -> overlap.
    """
    if len(pieces) < 2:
        return False
    # one small canvas around all the pieces (full-frame is overkill)
    xs, ys, xe, ye = [], [], [], []
    for c in pieces:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)
    x0, y0 = max(0, min(xs) - 2), max(0, min(ys) - 2)
    x1, y1 = max(xe) + 2, max(ye) + 2
    filled = []
    for c in pieces:
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1, offset=(-x0, -y0))
        filled.append(m)
    for i in range(len(pieces)):
        area_i = max(1.0, float(cv2.contourArea(pieces[i])))
        for j in range(len(pieces)):
            if i == j:
                continue
            inside = cv2.countNonZero(cv2.bitwise_and(filled[i], filled[j]))
            if inside > max_frac * area_i:
                return True
    return False


def detect_all_onions(bgr, gray, existing, median_area=None):
    """
    "Detect ALL onions" engine: extra passes that catch onions the main
    Otsu mask MISSED (low contrast, uneven light, pale skins).

    PASS 2 - adaptive threshold, BOTH polarities. Global Otsu uses ONE
             cut for the whole photo and fails in uneven lighting
             (half-shadowed tables). Adaptive thresholding cuts locally,
             so a pale onion in a shadow is still separated.
    PASS 3 - a Hough circle sweep over the whole frame for round,
             onion-sized objects the thresholds never separated.

    Every candidate is validated (roundish, not flat background, and
    roughly ONION-SIZED: at least MIN_ONION_FRAC of the median onion
    area when one is known, so textured-background fragments cannot
    sneak in) and de-duplicated against existing detections with bbox
    IoU, so the count can only GROW where something was genuinely
    missed.  `median_area` may be None (labeling tools that pass no
    existing detections) - then only the absolute size floor applies.
    Returns a list of extra contours (may be empty).
    """
    h, w = gray.shape
    extras = []
    used_boxes = [cv2.boundingRect(c) for c in existing]
    k = np.ones(MORPH_K, np.uint8)
    min_onion_px = (MIN_ONION_FRAC * median_area) if median_area else 0

    def overlaps(cx, cy, r):
        for (bx, by, bw, bh) in used_boxes:
            px = min(max(cx, bx), bx + bw)
            py = min(max(cy, by), by + bh)
            if (px - cx) ** 2 + (py - cy) ** 2 <= (0.9 * r) ** 2:
                return True
        return False

    def textured(cx, cy, r):
        # flat background (even lighting, no object) = not an onion
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return False
        return float(gray[y0:y1, x0:x1].std()) >= 5.0

    def accept(c, min_circ=0.35):
        a = cv2.contourArea(c)
        if a < MIN_AREA_PX or a < min_onion_px or a > 0.5 * w * h:
            return          # speck / background texture / the whole frame
        if circularity(c) < min_circ:
            return                      # onions are round-ish
        bb = cv2.boundingRect(c)
        if any(_bbox_iou(bb, ub) > 0.3 for ub in used_boxes):
            return                      # already found by another pass
        x, y, bw2, bh2 = bb
        if not textured(x + bw2 // 2, y + bh2 // 2,
                        max(8, min(bw2, bh2) // 2)):
            return
        extras.append(c)
        used_boxes.append(bb)

    # ---- PASS 2: adaptive threshold, both polarities ----
    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        ad = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   polarity, 51, 7)
        ad = cv2.morphologyEx(ad, cv2.MORPH_OPEN, k, iterations=2)
        ad = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(ad, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            accept(c)

    # ---- PASS 3: Hough circle sweep ----
    blur2 = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blur2, cv2.HOUGH_GRADIENT, dp=1.5,
                               minDist=max(30, int(w * 0.05)),
                               param1=110, param2=30,
                               minRadius=max(12, int(w * 0.03)),
                               maxRadius=int(w * 0.16))
    if circles is not None:
        for cx, cy, r in circles[0][:40]:
            cx, cy, r = int(cx), int(cy), int(r)
            if overlaps(cx, cy, r):
                continue
            if not textured(cx, cy, r):
                continue
            disk = np.zeros((h, w), np.uint8)
            cv2.circle(disk, (cx, cy), r, 255, -1)
            cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                accept(cnts[0], min_circ=0.2)   # a disk is round by design
    return extras


def exif_focal_px(image_path, image_width_px):
    """
    NO-COIN scale estimation from optics (real photogrammetry).

    Lens formula:  object_mm / distance_mm = size_px / focal_px
        ->  px_per_mm = focal_px / distance_mm      (caller divides!)
    focal_px comes from the photo's EXIF 35mm-equivalent focal length
    (a 35mm frame is 36 mm wide):  focal_px = F35 / 36 * width_px.

    The user types the camera distance (e.g. 40 cm above the table).
    The distance guess dominates, so expect about +/-20 percent.
    Returns (focal_px, None) or (None, reason).
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS
        pil = PILImage.open(image_path)
        exif = pil._getexif() if hasattr(pil, "_getexif") else None
        if not exif:
            return None, "photo has no EXIF data"
        f35 = None
        for k, v in exif.items():
            if TAGS.get(k, "") == "FocalLengthIn35mmFilm":
                try:
                    f35 = float(v)
                except (TypeError, ValueError):
                    f35 = None
                break
        if not f35:
            return None, "no 35mm-equivalent focal length in EXIF"
        return f35 / 36.0 * float(image_width_px), None
    except Exception as exc:
        return None, str(exc)


# ------------------------------------------------------------------
# STEP 8 - find the coin (the ruler of the photo)
# ------------------------------------------------------------------
def find_coin(blobs, hsv, median_area):
    """
    A coin is: very round (circ >= 0.80), small (area < 0.6 x median)
    AND METAL - clearly less colourful than the onion skins in the same
    photo. Without the colour gate a photo with NO coin "eats" its
    smallest onion as a fake coin (a small onion is round and small
    too), which silently poisons the mm scale.
    If several blobs qualify, pick the LEAST colourful one - a coin is
    metallic gray (low saturation), onions are warm-coloured.
    Returns a dict, or None if no coin was found.
    """
    cands = [c for c in blobs
             if circularity(c) >= COIN_CIRC_MIN
             and cv2.contourArea(c) < COIN_AREA_MAX * median_area]
    if not cands:
        return None

    def mean_saturation(c):
        m = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        return float(cv2.mean(hsv[:, :, 1], m)[0])

    sats = [mean_saturation(c) for c in cands]
    med_sat = float(np.median([mean_saturation(c) for c in blobs]))
    kept = [(c, s) for c, s in zip(cands, sats)
            if s <= COIN_SAT_MAX or s <= 0.45 * med_sat]
    if not kept:
        return None                     # nothing metallic -> no coin
    kept.sort(key=lambda t: t[1])
    coin = kept[0][0]
    (cx, cy), r = cv2.minEnclosingCircle(coin)
    return {"contour": coin, "center": (int(cx), int(cy)),
            "d_px": 2.0 * float(r)}


# ------------------------------------------------------------------
# STEP 9 - color features, measured INSIDE one onion's mask only
# ------------------------------------------------------------------
def onion_features(gray, hsv, mask):
    """
    Returns fractions (0..1) of the onion's visible surface that are:
      green  - a green shoot (sprout)
      brown  - DARK brown skin (defects are darker than healthy skin)
      dark   - very dark gray (rot / deep damage)
    """
    m = mask > 0
    total = int(m.sum())
    if total == 0:
        return {"green": 0.0, "brown": 0.0, "dark": 0.0}
    H = hsv[:, :, 0][m].astype(np.int32)
    S = hsv[:, :, 1][m].astype(np.int32)
    V = hsv[:, :, 2][m].astype(np.int32)
    G = gray[m].astype(np.int32)
    green = float(np.mean((H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)))
    brown = float(np.mean((H >= 8) & (H <= 25) & (S >= 60) & (V <= 160)))
    dark = float(np.mean(G < 70))
    return {"green": green, "brown": brown, "dark": dark}


# ------------------------------------------------------------------
# STEP 10 - classify one onion (rule based)
# ------------------------------------------------------------------
def classify(feats, d_mm, full_visible=True):
    """Order matters: check worst problems first.
    Size rules (UNDERSIZED) only apply to FULLY VISIBLE onions - a
    partly hidden onion always measures too small, that would be unfair."""
    if feats["green"] >= GREEN_SPROUTED:
        return "SPROUTED"
    if feats["dark"] >= DARK_ROTTEN or feats["brown"] >= BROWN_ROTTEN:
        return "ROTTEN"
    if feats["dark"] >= DARK_DAMAGED or feats["brown"] >= BROWN_DAMAGED:
        return "DAMAGED"
    if full_visible and d_mm < UNDERSIZED_MM:
        return "UNDERSIZED"
    return "GOOD"


# ------------------------------------------------------------------
# STEP 11 - grade one onion
# ------------------------------------------------------------------
def grade_of(label, d_mm, full_visible=True):
    """Only GOOD onions can get a grade. Anything else is REJECT.
    A partly-hidden GOOD onion gets CHECK: its visible part looks fine
    but its size cannot be measured fairly (diameter is a lower bound)."""
    if label != "GOOD":
        return "REJECT"
    if not full_visible:
        return "CHECK"
    if GRADE_A_MM[0] <= d_mm <= GRADE_A_MM[1]:
        return "A"
    if URS_MM[0] <= d_mm <= URS_MM[1]:
        return "URS"
    return "REJECT"


# ------------------------------------------------------------------
# Report writers (STEP 12)
# ------------------------------------------------------------------
def _font(size, bold=False):
    """Pick a font that exists on this computer (or a safe fallback)."""
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "arialbd.ttf" if bold else "arial.ttf"]
    folders = ["/usr/share/fonts/truetype/dejavu/",
               "C:/Windows/Fonts/",
               "/System/Library/Fonts/Supplemental/"]
    for n in names:
        for f in folders:
            try:
                return ImageFont.truetype(f + n, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_annotated(rep, bgr, coin, path):
    """The photo again, but with boxes, labels and sizes drawn on it.
    Labels are drawn left-to-right and pushed down when they would
    overlap, so every onion stays readable."""
    canvas = bgr.copy()
    label_jobs = []

    # coin marker
    if coin is not None:
        cv2.circle(canvas, coin["center"], int(coin["d_px"] / 2),
                   (255, 255, 0), 2)
        label_jobs.append((f"COIN {rep['coin_mm']:g} mm",
                           coin["center"][0] - 40,
                           coin["center"][1] - int(coin["d_px"] / 2) - 22,
                           (255, 255, 0)))

    # per-onion box + outline
    for o in rep["onions"]:
        x, y, w, h = o["bbox"]
        color = CLASS_COLORS[o["label"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.drawContours(canvas, [o["contour"]], -1, color, 2)
        lay = o.get("layer", "L1")
        lay_txt = "" if lay == "L1" else f" [{lay}]"
        label_jobs.append(
            (f"#{o['id']} {o['label']} {o['diameter_mm']:.0f}mm "
             f"{o['grade']}{lay_txt}",
             x, y, color))

    # draw all labels left-to-right, avoiding overlaps
    label_jobs.sort(key=lambda j: (j[1], j[2]))
    used = []
    for text, lx, ly, color in label_jobs:
        lx, ly = max(0, lx), max(0, ly)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.55, 1)
        rw, rh = tw + 8, th + base + 6
        rect = [lx, ly, lx + rw, ly + rh]

        def hits(a, b):   # do two rectangles overlap?
            return not (a[2] < b[0] or b[2] < a[0]
                        or a[3] < b[1] or b[3] < a[1])

        tries = 0
        while any(hits(rect, u) for u in used) and tries < 25:
            ly += rh + 4                       # step down until free
            rect = [lx, ly, lx + rw, ly + rh]
            tries += 1
        used.append(rect)
        cv2.rectangle(canvas, (lx, ly), (lx + rw, ly + rh), (30, 30, 30), -1)
        cv2.putText(canvas, text, (lx + 4, ly + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # summary strip on top
    g, p = rep["grade_counts"], rep["grade_percent"]
    wtxt = (f"   ~{rep['estimated_weight_kg']:.2f}kg"
            if rep["onion_count"] else "")
    summary = (f"Batch {rep['batch_id']}   onions: {rep['onion_count']}   "
               f"A: {p['A']:.0f}%  URS: {p['URS']:.0f}%  "
               f"REJ: {p['REJECT']:.0f}%{wtxt}")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (30, 90, 40), -1)
    cv2.putText(canvas, summary, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # honesty strip at the bottom
    strip_y = canvas.shape[0] - 26
    cv2.rectangle(canvas, (0, strip_y), (canvas.shape[1], canvas.shape[0]),
                  (0, 215, 255), -1)
    cv2.putText(canvas, "Visible surface analysis only - cannot detect "
                "internal rot / damage / moisture",
                (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 60, 160), 1, cv2.LINE_AA)
    if path:
        cv2.imwrite(path, canvas)
    return canvas


def put_label(canvas, text, x, y, color):
    """Small text with a dark background so it is easy to read.
    (Used by other scripts that import grader.)"""
    x = max(0, x)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_top = max(0, y - th - base - 6)
    cv2.rectangle(canvas, (x, y_top), (x + tw + 8, y_top + th + base + 6),
                  (30, 30, 30), -1)
    cv2.putText(canvas, text, (x + 4, y_top + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _font_mono(size):
    """Monospace font for table-like text in report images."""
    names = ["DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc"]
    folders = ["/usr/share/fonts/truetype/dejavu/",
               "C:/Windows/Fonts/",
               "/System/Library/Fonts/Supplemental/"]
    for n in names:
        for f in folders:
            try:
                return ImageFont.truetype(f + n, size)
            except Exception:
                pass
    return _font(size)


def _wrap_text(text, font, width):
    """Simple word-wrap for PIL text drawing."""
    lines, cur = [], ""
    for word in str(text).split():
        trial = (cur + " " + word).strip()
        if font.getlength(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def make_full_report_jpg(rep, bgr, coin, path):
    """
    ONE shareable JPEG that contains the WHOLE report:
    header (batch/time/count/weight) + the annotated photo + grade bars
    + quantity + layer-by-layer + per-onion table + flags + summary +
    honesty disclaimer. Perfect for WhatsApp / printing.
    """
    Wc, mg = 1150, 24
    inner = Wc - 2 * mg

    ann = make_annotated(rep, bgr, coin, None)
    ann = cv2.resize(ann, (inner, int(ann.shape[0] * inner / ann.shape[1])),
                     interpolation=cv2.INTER_AREA)
    ann_pil = Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

    f_disp, f_h = _font(30, True), _font(18, True)
    f_t, f_m, f_s = _font(15), _font_mono(14), _font(12)

    g, p = rep["grade_counts"], rep["grade_percent"]
    cc = rep["class_counts"]
    la = rep.get("layer_analysis") or {"layers": []}
    n = rep["onion_count"]
    flags = list(rep.get("quality_flags") or []) + list(rep.get("warnings") or [])
    sum_lines = _wrap_text(rep.get("summary", ""), f_t, inner - 36)

    # ---- measure total height ----
    Hc = 88                                   # header
    Hc += ann_pil.height + 16                 # photo
    Hc += 34 + 4 * 62 + 8                     # grade bars (A/URS/REJ/CHECK)
    Hc += 30 + 4 * 30 + 14                    # quantity chips
    if la["layers"]:
        Hc += 46 + 34 * len(la["layers"]) + 42
    Hc += 50 + 30 * max(n, 1) + 14            # per-onion table
    if flags:
        Hc += 36 + 26 * len(flags)
    Hc += 46 + 24 * len(sum_lines)            # summary
    Hc += 84                                  # disclaimer

    card = Image.new("RGB", (Wc, Hc), (250, 250, 250))
    d = ImageDraw.Draw(card)
    y = 0

    # header (dark, inverted section)
    d.rectangle([0, 0, Wc, 88], fill=(15, 23, 42))
    d.text((mg, 14), "ONION BATCH - FULL QUALITY REPORT",
           font=f_disp, fill=(255, 255, 255))
    d.text((mg, 56),
           f"{rep['batch_id']}   |   {rep['timestamp']}   |   "
           f"{n} onions   |   ~{rep['estimated_weight_kg']} kg",
           font=f_m, fill=(147, 197, 253))
    y = 88

    # annotated photo
    card.paste(ann_pil, (mg, y + 8))
    d.rectangle([mg - 1, y + 7, mg + inner, y + 8 + ann_pil.height],
                outline=(226, 232, 240), width=1)
    y += ann_pil.height + 16

    # grade bars
    d.text((mg, y + 4), "GRADE RESULTS", font=f_h, fill=(15, 23, 42))
    y += 34
    bars = [("GRADE A (45-65 mm)", g.get("A", 0), p.get("A", 0.0),
             (21, 128, 61), (34, 197, 94)),
            ("GRADE URS (35-70 mm)", g.get("URS", 0), p.get("URS", 0.0),
             (180, 83, 9), (245, 158, 11)),
            ("REJECT", g.get("REJECT", 0), p.get("REJECT", 0.0),
             (185, 28, 28), (239, 68, 68)),
            ("CHECK (partly hidden)", g.get("CHECK", 0), p.get("CHECK", 0.0),
             (51, 65, 85), (100, 116, 139))]
    for name, cnt, pct, c1, c2 in bars:
        bw = int(inner * 0.62 * max(pct, 1.5) / 100.0)
        d.rounded_rectangle([mg, y + 2, mg + inner, y + 46], radius=8,
                            fill=(241, 245, 249))
        if pct > 0 or cnt > 0:
            d.rounded_rectangle([mg, y + 2, mg + bw, y + 46], radius=8, fill=c1)
        d.text((mg + 12, y + 10), name, font=f_t, fill=(255, 255, 255)
               if (pct > 0 or cnt > 0) else (100, 116, 139))
        d.text((mg + inner - 190, y + 8),
               f"{cnt} onions  |  {pct}%", font=f_h,
               fill=(51, 65, 85) if (pct > 0 or cnt > 0) else (148, 163, 184))
        y += 62
    y += 8

    # quantity block
    d.text((mg, y + 2), "QUANTITY (estimated)", font=f_h, fill=(15, 23, 42))
    y += 30
    q_items = [
        f"total weight ~ {rep['estimated_weight_kg']} kg  (~{rep['bags_50kg']} x 50-kg bags)",
        f"onions cover {rep['coverage_percent']}% of the photo",
        f"weight model: mass_g = {rep['weight_k']} x diameter_mm^3  (calibrate once with a scale)",
        f"scale: {rep['px_per_mm']} px/mm  [{rep['scale_source']}]"]
    for q in q_items:
        d.ellipse([mg + 2, y + 8, mg + 8, y + 14], fill=(0, 82, 255))
        d.text((mg + 18, y + 2), q, font=f_t, fill=(51, 65, 85))
        y += 30
    y += 14

    # layer-by-layer
    if la["layers"]:
        d.text((mg, y + 2), "LAYER-BY-LAYER (pile depth, occlusion estimate)",
               font=f_h, fill=(15, 23, 42))
        y += 40
        for L in la["layers"]:
            d.rounded_rectangle([mg, y, mg + inner, y + 28], radius=6,
                                fill=(219, 234, 254))
            d.text((mg + 10, y + 5),
                   f"{L['layer']}  {L['name']}:  {L['count']} onions | "
                   f"A {L['grade_percent']['A']}% | URS {L['grade_percent']['URS']}% | "
                   f"REJ {L['grade_percent']['REJECT']}% | CHECK {L['grade_percent'].get('CHECK', 0)}% "
                   f"| ~{L['est_weight_kg']} kg | avg {L['avg_diameter_mm']} mm",
                   font=f_m, fill=(30, 58, 138))
            y += 34
        d.text((mg, y + 4), "i " + la.get("note", ""), font=f_s, fill=(100, 116, 139))
        y += 42

    # per-onion table
    d.text((mg, y + 2), "PER-ONION DETAIL", font=f_h, fill=(15, 23, 42))
    y += 36
    cols = [mg + 4, mg + 46, mg + 210, mg + 330, mg + 430, mg + 530,
            mg + 640, mg + 750, mg + 860]
    heads = ["#", "class", "diameter", "grade", "layer", "green%", "brown%",
             "dark%", "vis%"]
    d.rectangle([mg, y - 2, mg + inner, y + 24], fill=(15, 23, 42))
    for cx, htxt in zip(cols, heads):
        d.text((cx, y + 3), htxt, font=f_m, fill=(255, 255, 255))
    y += 30
    for i, o in enumerate(rep["onions"]):
        if i % 2 == 0:
            d.rectangle([mg, y - 2, mg + inner, y + 24], fill=(241, 245, 249))
        f = o["features"]
        row = [str(o["id"]), o["label"], f"{o['diameter_mm']} mm", o["grade"],
               o.get("layer", "-"), f"{100*f['green']:.1f}",
               f"{100*f['brown']:.1f}", f"{100*f['dark']:.1f}",
               f"{100*o.get('visibility', 1):.0f}"]
        for cx, val in zip(cols, row):
            d.text((cx, y + 2), val, font=f_m, fill=(30, 41, 59))
        y += 30
    y += 14

    # flags
    if flags:
        d.text((mg, y + 2), "IMAGE QUALITY FLAGS", font=f_h, fill=(180, 83, 9))
        y += 36
        for fl in flags:
            d.text((mg + 6, y), "! " + str(fl), font=f_t, fill=(146, 64, 14))
            y += 26

    # summary
    d.text((mg, y + 2), "SUMMARY", font=f_h, fill=(15, 23, 42))
    y += 40
    for ln in sum_lines:
        d.text((mg + 6, y), ln, font=f_t, fill=(51, 65, 85))
        y += 24

    # disclaimer footer
    d.rectangle([0, Hc - 84, Wc, Hc], fill=(255, 247, 237))
    d.text((mg, Hc - 74),
           "VISIBLE SURFACE ANALYSIS ONLY - a normal photo cannot detect internal rot, "
           "internal damage or internal moisture.", font=f_s, fill=(124, 45, 18))
    d.text((mg, Hc - 52),
           "Demo thresholds - real accuracy must be measured on a labeled test set. "
           "Weight is a calibratable estimate.", font=f_s, fill=(124, 45, 18))
    d.text((mg, Hc - 30),
           "OnionGrader - SIH26031", font=f_s, fill=(150, 100, 60))
    if path:
        card.save(path, quality=90)
    return card


def make_report_card(rep, annotated_path, path):
    """A friendly one-picture 'report card' of the whole batch (PIL)."""
    Wc, Hc = 1150, 700
    card = Image.new("RGB", (Wc, Hc), (246, 244, 238))
    d = ImageDraw.Draw(card)

    d.rectangle([0, 0, Wc, 84], fill=(22, 86, 42))
    d.text((24, 16), "ONION QUALITY REPORT CARD",
           font=_font(34, True), fill=(255, 255, 255))
    d.text((26, 56), "AI onion grading  -  SIH26031",
           font=_font(16), fill=(196, 226, 200))

    # left side: the annotated photo
    try:
        ann = Image.open(annotated_path)
        ann.thumbnail((520, 500))
        card.paste(ann, (24, 108))
        d.rectangle([23, 107, 25 + ann.width, 109 + ann.height],
                    outline=(150, 150, 150), width=1)
    except Exception:
        d.text((24, 108), "(annotated image not available)",
               font=_font(16), fill=(150, 0, 0))

    # right side: the numbers
    x0 = 570
    d.text((x0, 104), f"Batch ID : {rep['batch_id']}",
           font=_font(20, True), fill=(30, 30, 30))
    d.text((x0, 134), f"Date     : {rep['timestamp']}",
           font=_font(18), fill=(50, 50, 50))
    img_name = rep["image"]
    if len(img_name) > 42:
        img_name = "..." + img_name[-39:]
    d.text((x0, 162), f"Photo    : {img_name}",
           font=_font(18), fill=(50, 50, 50))
    d.text((x0, 190), f"Scale    : {rep['px_per_mm']:.2f} px/mm "
                      f"({rep['scale_source']})", font=_font(18), fill=(50, 50, 50))
    d.text((x0, 218), f"Onions : {rep['onion_count']}   |   watershed splits: "
                      f"{rep['watershed_splits']}   |   est. weight ~ "
                      f"{rep['estimated_weight_kg']} kg",
           font=_font(18), fill=(50, 50, 50))

    rows = [("GRADE A",   rep["grade_percent"]["A"],      rep["grade_counts"]["A"],      (24, 128, 56)),
            ("GRADE URS", rep["grade_percent"]["URS"],    rep["grade_counts"]["URS"],    (198, 124, 0)),
            ("REJECT",    rep["grade_percent"]["REJECT"], rep["grade_counts"]["REJECT"], (186, 34, 34))]
    y = 254
    for name, pct, cnt, col in rows:
        d.rounded_rectangle([x0, y, Wc - 24, y + 66], radius=10, fill=col)
        d.text((x0 + 16, y + 10), name, font=_font(24, True), fill=(255, 255, 255))
        d.text((x0 + 16, y + 38), f"{cnt} of {rep['onion_count']} onions",
               font=_font(15), fill=(255, 255, 255))
        d.text((Wc - 190, y + 14), f"{pct:.1f}%", font=_font(30, True),
               fill=(255, 255, 255))
        y += 78

    cc = rep["class_counts"]
    d.text((x0, y + 2), "Visible surface classes:",
           font=_font(16, True), fill=(60, 60, 60))
    d.text((x0, y + 26),
           (f"GOOD {cc['GOOD']}   DAMAGED {cc['DAMAGED']}   ROTTEN {cc['ROTTEN']}   "
            f"SPROUTED {cc['SPROUTED']}   UNDERSIZED {cc['UNDERSIZED']}"),
           font=_font(15), fill=(60, 60, 60))
    la = rep.get("layer_analysis") or {}
    if la.get("layers"):
        lay_s = "   ".join(f"{L['layer']} {L['count']}" for L in la["layers"])
        d.text((x0, y + 48), "Pile layers (estimate): " + lay_s,
               font=_font(15), fill=(0, 82, 255))
        flag_y = y + 70
    else:
        flag_y = y + 48
    if rep.get("quality_flags"):
        d.text((x0, flag_y), "Flags: " + "; ".join(rep["quality_flags"][:2]),
               font=_font(13), fill=(150, 80, 0))

    d.rectangle([0, Hc - 62, Wc, Hc], fill=(255, 233, 181))
    d.text((16, Hc - 54), "VISIBLE SURFACE ANALYSIS ONLY - a normal photo cannot "
           "detect internal rot, internal damage or internal moisture.",
           font=_font(14), fill=(122, 62, 0))
    d.text((16, Hc - 32), "Demo thresholds - accuracy must be measured on a "
           "labeled test set.", font=_font(14), fill=(122, 62, 0))
    card.save(path)


def make_text_report(rep, path):
    """A plain-text report a person can read in Notepad."""
    L = []
    L.append("=" * 64)
    L.append("ONION QUALITY REPORT")
    L.append("=" * 64)
    L.append(f"batch id    : {rep['batch_id']}")
    L.append(f"date/time   : {rep['timestamp']}")
    L.append(f"photo       : {rep['image']}")
    L.append(f"scale       : {rep['px_per_mm']:.2f} px/mm  [{rep['scale_source']}]")
    L.append(f"onions found: {rep['onion_count']}")
    L.append(f"watershed splits (touching pairs): {rep['watershed_splits']}")
    L.append("-" * 64)
    L.append("GRADE SUMMARY")
    for k in ("A", "URS", "REJECT", "CHECK"):
        L.append(f"  {k:<7}: {rep['grade_counts'][k]:3d}  "
                 f"({rep['grade_percent'][k]:5.1f}%)")
    L.append("-" * 64)
    L.append("ESTIMATED QUANTITY")
    L.append(f"  total weight : ~ {rep['estimated_weight_kg']} kg "
             f"(~ {rep['bags_50kg']} x 50-kg bags)")
    L.append(f"  coverage     : onions fill {rep['coverage_percent']}% "
             "of the photo")
    L.append(f"  weight model : mass_g = {rep['weight_k']} x diameter_mm^3 "
             "(calibrate once with a scale)")
    if rep["quality_flags"]:
        L.append("-" * 64)
        L.append("IMAGE QUALITY FLAGS")
        for f in rep["quality_flags"]:
            L.append(f"  ! {f}")
    L.append("-" * 64)
    L.append("LAYER-BY-LAYER ANALYSIS (occlusion estimate)")
    la = rep.get("layer_analysis") or {}
    for lay in la.get("layers", []):
        L.append(f"  {lay['layer']:<3}{lay['name']:<30}: {lay['count']:3d} onions | "
                 f"A {lay['grade_percent']['A']:5.1f}%  URS {lay['grade_percent']['URS']:5.1f}%"
                 f"  REJ {lay['grade_percent']['REJECT']:5.1f}% | "
                 f"~{lay['est_weight_kg']} kg | avg {lay['avg_diameter_mm']} mm")
    if la.get("layers"):
        L.append(f"  NOTE: {la['note']}")
    L.append("-" * 64)
    L.append("SUMMARY")
    L.append(f"  {rep['summary']}")
    L.append("-" * 64)
    L.append("PER-ONION DETAIL")
    L.append(f"  {'#':<3}{'class':<12}{'diameter':>9}  {'grade':<7}{'layer':<6}"
             f"green%  brown%  dark%")
    for o in rep["onions"]:
        f = o["features"]
        L.append(f"  {o['id']:<3}{o['label']:<12}{o['diameter_mm']:>7.1f}mm "
                 f"  {o['grade']:<7}{o.get('layer', '-'):<6}{100*f['green']:5.1f}  "
                 f"{100*f['brown']:5.1f}  {100*f['dark']:5.1f}")
    L.append("-" * 64)
    for w in rep["warnings"]:
        L.append(f"WARNING: {w}")
    L.append(f"NOTE: {DISCLAIMER}")
    L.append("=" * 64)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ------------------------------------------------------------------
# NEW: pile LAYER detection (top / middle / buried) - occlusion based
# ------------------------------------------------------------------
def compute_visibility(contours):
    """
    For every onion: EXTENT = area / convex-hull area.

    Why this is the right occlusion cue: a fully visible onion is a full
    disc, so its area almost fills its convex hull (extent ~0.9-1.0).
    An onion lying lower in the pile is covered by others, so the camera
    only sees a crescent - its hull stays as big as the whole onion but
    the visible area shrinks (extent 0.2-0.6). Works for merged AND
    watershed-split regions, and does not depend on onion size.
    """
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        out.append(area / hull_area if hull_area > 0 else 1.0)
    return out


def layer_of(visibility):
    """Map a visibility score to a pile layer (see LAYER_BANDS)."""
    for t, key, _name in LAYER_BANDS:
        if visibility >= t:
            return key
    return "L3"


def build_layer_analysis(onions):
    """Group onions by layer and compute per-layer statistics."""
    layers = []
    for _t, key, name in LAYER_BANDS:
        grp = [o for o in onions if o.get("layer") == key]
        if not grp:
            continue
        gcounts = {"A": sum(1 for o in grp if o["grade"] == "A"),
                   "URS": sum(1 for o in grp if o["grade"] == "URS"),
                   "REJECT": sum(1 for o in grp if o["grade"] == "REJECT"),
                   "CHECK": sum(1 for o in grp if o["grade"] == "CHECK")}
        gpct = {k: round(v * 100.0 / len(grp), 1) for k, v in gcounts.items()}
        layers.append({
            "layer": key, "name": name, "count": len(grp),
            "grade_counts": gcounts, "grade_percent": gpct,
            "est_weight_kg": round(sum(o["mass_g"] for o in grp) / 1000.0, 2),
            "avg_diameter_mm": round(float(np.mean([o["diameter_mm"]
                                                    for o in grp])), 1)})
    return {"layers": layers, "note": LAYER_NOTE}


# ------------------------------------------------------------------
# NEW: quantity estimation + image quality checks + plain summary
# ------------------------------------------------------------------
def weight_of(d_mm, weight_k=WEIGHT_K_G_PER_MM3):
    """Estimated mass of ONE onion from its diameter (grams).
    Rough but useful: mass grows with the cube of the diameter.
    CALIBRATE the constant once with a kitchen scale (see settings)."""
    return weight_k * (d_mm ** 3)


def quality_checks(gray, hsv):
    """Honest 'instrument self-test': flag bad photos instead of
    silently returning bad numbers. Returns a list of flag strings."""
    flags = []
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 60:
        flags.append(f"photo looks blurry (sharpness {lap_var:.0f} < 60) "
                     "- hold the camera steadier / refocus")
    mean_v = float(np.mean(hsv[:, :, 2]))
    if mean_v < 60:
        flags.append("image is very dark - move to better light")
    glare = float(np.mean(hsv[:, :, 2] > 250))
    if glare > 0.10:
        flags.append(f"glare/overexposure on {100 * glare:.0f}% of pixels "
                     "- avoid direct flash")
    return flags


def build_summary(n, splits, gc, gp, cc, weight_kg, scale_source, flags):
    """One human sentence describing the batch (shown in reports)."""
    if n == 0:
        return ("No onions detected. Check the light and background - "
                "onions must stand out from what is behind them.")
    parts = [f"{n} onion{'s' if n != 1 else ''} detected"]
    if splits:
        parts.append(f"{splits} touching group{'s' if splits != 1 else ''} "
                     "split apart")
    reasons = []
    for key, word in (("ROTTEN", "rotten"), ("DAMAGED", "damaged"),
                      ("SPROUTED", "sprouted"), ("UNDERSIZED", "undersized")):
        if cc[key]:
            reasons.append(f"{cc[key]} {word}")
    line = (f"{gc['A']} Grade A ({gp['A']}%), {gc['URS']} URS "
            f"({gp['URS']}%), {gc['REJECT']} reject")
    if gc.get("CHECK"):
        line += f", {gc['CHECK']} partly hidden (size: to check)"
    if reasons:
        line += " - main issues: " + ", ".join(reasons)
    parts.append(line)
    if weight_kg:
        parts.append(f"estimated total weight about {weight_kg:.2f} kg")
    parts.append(f"size scale from {scale_source}")
    if flags:
        parts.append(f"{len(flags)} image-quality flag"
                     f"{'s' if len(flags) != 1 else ''} raised - see flags")
    return ". ".join(parts) + "."


# ------------------------------------------------------------------
# THE MAIN FUNCTION - put the whole pipeline together
# ------------------------------------------------------------------
def analyze(image, coin_mm=27.0, batch_id=None, out_dir="outputs",
            weight_k=WEIGHT_K_G_PER_MM3, distance_mm=None, assume_mm=None,
            coin_assumed=False):
    """Run the full pipeline on one photo. Returns the report as a dict.

    `image` can be a FILE PATH or an already-loaded BGR image (numpy
    array) - the numpy version is used for LIVE camera frames.
    Pass out_dir=None to run fully in-memory (no report files written);
    that is what the live mode calls every second.
    """
    in_memory = isinstance(image, np.ndarray)
    if in_memory:
        bgr = _fit_width(image)
        stem, image_name = "live_frame", "live camera frame"
    else:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image))[0]
        bgr = read_image(image)
        image_name = image

    gray, mask = make_object_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if out_dir and not in_memory:
        cv2.imwrite(os.path.join(out_dir, f"debug_mask_{stem}.jpg"), mask)

    blobs = get_blobs(mask)
    # --- "detect ALL onions" guard for UNEVEN LIGHTING ---
    # If nothing was found, or one blob swallows > 25% of the photo
    # (a shadow region merged with the onions), re-threshold on an
    # illumination-FLATTENED copy of the image. Shadows disappear,
    # pale onions pop out.
    img_area = bgr.shape[0] * bgr.shape[1]
    suspicious = (not blobs) or (max(cv2.contourArea(c) for c in blobs)
                                 > 0.25 * img_area)
    if suspicious:
        # local-contrast pass: catches dark AND bright onions under
        # uneven lighting (global Otsu merges one of them into the bg)
        mask_lc = local_contrast_mask(gray, hsv)
        cand = get_blobs(mask_lc)
        deduped, boxes = [], []
        for c in cand:
            a = cv2.contourArea(c)
            if a < MIN_AREA_PX or a > 0.25 * img_area:
                continue
            bb = cv2.boundingRect(c)
            if any(_bbox_iou(bb, ub) > 0.3 for ub in boxes):
                continue
            deduped.append(c)
            boxes.append(bb)
        if deduped:   # refine candidates; the 'saner' check decides
            # refine each candidate to its true boundary with a local
            # Otsu cut, then de-duplicate once more
            refined, rboxes = [], []
            for c in deduped:
                rc = refine_local_otsu(gray, c, hsv=hsv)
                a = cv2.contourArea(rc)
                if a < MIN_AREA_PX:
                    continue
                bb = cv2.boundingRect(rc)
                if any(_bbox_iou(bb, ub) > 0.3 for ub in rboxes):
                    continue
                refined.append(rc)
                rboxes.append(bb)
            # take over when at least as many objects were found AND
            # they explain much less area (less shadow-garbage)
            saner = (len(refined) >= len(blobs) and
                     sum(cv2.contourArea(c) for c in refined) <
                     0.8 * sum(cv2.contourArea(c) for c in blobs))
            if saner:
                mask, blobs = mask_lc, refined
    warnings, watershed_splits = [], 0

    # --- scene suitability (honest diagnostics, not fake accuracy) ---
    # Real-world photos (heaps on jute sacks, textured surfaces, busy
    # backgrounds) break single-threshold segmentation. Instead of
    # silently returning nonsense numbers, TELL the user the SCENE is
    # the problem and what to change.
    coverage = 100.0 * float(mask.mean()) / 255.0
    if blobs and coverage > 55.0:
        warnings.append(
            f"objects cover {coverage:.0f}% of the photo and merge into "
            "big blobs - this looks like a HEAP or a busy/textured "
            "background. For reliable counting, spread the onions in "
            "ONE layer on a plain, contrasting surface.")

    # heap-detection flags (set inside the blobs block below; defined
    # here so the code AFTER the block can always read them)
    heap_mode = heap_estimated = False

    if blobs:
        areas = [cv2.contourArea(c) for c in blobs]
        median_area = float(np.median(areas))

        # --- HEAP CASE: one merged blob covers a big part of the photo
        # (a pile/heap of many onions). The median-area trick degenerates
        # here - the median of ONE blob is the blob itself, so
        # expected=1 and no split is ever attempted ("1 onion" for a
        # heap of 20). We estimate the onion radius from the blob's own
        # geometry instead and let watershed/Hough find the individuals.
        heap_mode = (len(blobs) == 1
                     and areas[0] > 0.25 * img_area)

        # --- split touching onions ---
        final, final_vis = [], []   # final_vis: None = compute extent later
        for c in blobs:
            area_c = cv2.contourArea(c)
            big = area_c > MERGED_FACTOR * median_area or heap_mode
            not_round = circularity(c) < CIRC_SPLIT_MAX
            expected = max(1, int(math.ceil(area_c / median_area)))
            r_hint = None
            if heap_mode:
                bx, by, bw, bh = cv2.boundingRect(c)
                # ~6 onions fit across the blob's shorter side -> radius
                r_hint = max(15.0, min(bw, bh) / 6.0)
                expected = max(2, int(math.ceil(
                    area_c / (math.pi * r_hint * r_hint))))
            # "underfull": the blob is too small for `expected` full
            # onions -> part of them must be HIDDEN (a pile!)
            underfull = area_c < 0.85 * expected * median_area
            if big and (not_round or underfull or heap_mode):
                pieces, pieces_vis = [], []
                if expected >= 2:
                    pieces = watershed_split(bgr, c, expected=expected)
                    if (len(pieces) == expected
                            and not _pieces_overlap(pieces)):
                        watershed_splits += expected - 1
                        pieces_vis = [None] * len(pieces)   # keep lists aligned!
                    else:
                        # no clean split: either watershed failed, or its
                        # "pieces" overlap (one onion mis-split into a
                        # nested ring+core pair) -> pretend it did not split
                        pieces, pieces_vis = [], []
                if len(pieces) < 2:
                    # rescue: Hough circle search for hidden onions
                    # (r_hint only in heap mode - see hough_split docs)
                    hpieces, hvis = hough_split(bgr, mask, c, median_area,
                                                r_hint=r_hint)
                    if (len(hpieces) >= 2
                            and not _pieces_overlap(hpieces)):
                        pieces = hpieces
                        pieces_vis = hvis
                        watershed_splits += len(pieces) - 1
                        if heap_mode:
                            heap_estimated = True
                if len(pieces) >= 2:
                    final += pieces
                    final_vis += pieces_vis
                    if heap_mode:
                        # heap split by EITHER watershed or Hough ->
                        # the count is still only an estimate
                        heap_estimated = True
                else:
                    final.append(c)
                    final_vis.append(None)
            else:
                final.append(c)
                final_vis.append(None)
        # --- "detect ALL onions": secondary passes for missed ones ---
        extras = detect_all_onions(bgr, gray, final, median_area)
        if extras:
            final += extras
            final_vis += [None] * len(extras)
            warnings.append(
                f"{len(extras)} onion(s) found by the SECONDARY detector "
                "(low contrast / uneven light) - check their boxes on the "
                "annotated photo")

        # any region still without a visibility value (Hough leftovers,
        # secondary-detector finds): use the convex-hull extent
        need = [i for i, v in enumerate(final_vis) if v is None]
        if need:
            for i, v in zip(need, compute_visibility([final[i] for i in need])):
                final_vis[i] = v
        # defensive: no visibility may ever be None from here on
        final_vis = [1.0 if v is None else v for v in final_vis]

        # --- coin ruler ---
        coin = find_coin(final, hsv, median_area)
        if coin is not None:
            px_per_mm = coin["d_px"] / coin_mm
            if coin_assumed:
                scale_source = (f"auto-detected coin, assumed Rs.10/Rs.2 "
                                f"({coin_mm:g} mm)")
            else:
                scale_source = f"coin {coin_mm:g} mm"
            # remove the coin from BOTH lists (they must stay aligned)
            for _idx, _c in enumerate(final):
                if _c is coin["contour"]:
                    del final[_idx]
                    del final_vis[_idx]
                    break
        else:
            # ---- NO COIN: honest estimate modes (no exact mm possible) ----
            ds = [2 * cv2.minEnclosingCircle(c)[1] for c in final]
            med_d_px = float(np.median(ds)) if ds else 1.0
            used_assume = float(assume_mm) if assume_mm else FALLBACK_ONION_MM
            if distance_mm:
                # mode A: optics estimate from the typed camera distance
                if in_memory:
                    f_px, err = None, "live frames carry no EXIF"
                else:
                    f_px, err = exif_focal_px(image_name, bgr.shape[1])
                if f_px and distance_mm > 0:
                    px_per_mm = f_px / float(distance_mm)
                    scale_source = (f"camera-distance estimate "
                                    f"({distance_mm:g} mm, EXIF focal)")
                    warnings.append(
                        "Sizes ESTIMATED from the camera distance you typed "
                        "(about +/-20%). Put a coin in the photo for exact "
                        "millimetres.")
                else:
                    px_per_mm = med_d_px / used_assume
                    scale_source = (f"assumed median onion {used_assume:g} mm "
                                    f"(EXIF unavailable: {err})")
                    warnings.append(
                        "Could not read the focal length from EXIF - sizes "
                        "GUESSED from the assumed standard size.")
            else:
                # mode B: assume a standard onion size
                px_per_mm = med_d_px / used_assume
                scale_source = (f"NO COIN - assumed median onion "
                                f"{used_assume:g} mm")
                warnings.append(
                    "No coin/reference found - sizes are ESTIMATES, not "
                    "measurements. Put a coin in the photo for exact mm.")

        # --- features + classify + grade, onion by onion ---
        # pile-layer cue for every final region (before building dicts)
        vis_list = final_vis

        onions = []
        for i, (c, vis) in enumerate(zip(final, vis_list), 1):
            omask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(omask, [c], -1, 255, -1)
            mbool = omask > 0
            feats = onion_features(gray, hsv, omask)
            (_, _), r = cv2.minEnclosingCircle(c)
            d_mm = (2.0 * float(r)) / px_per_mm
            vis = 1.0 if vis is None else float(vis)
            layer = layer_of(vis)
            full_visible = (layer == "L1")
            label = classify(feats, d_mm, full_visible=full_visible)
            onions.append({
                "id": i, "label": label,
                "diameter_mm": round(d_mm, 1),
                "grade": grade_of(label, d_mm, full_visible=full_visible),
                "layer": layer,
                "visibility": round(vis, 3),
                "diameter_note": (None if full_visible else
                                  "approximate - onion partly hidden"),
                "mass_g": round(weight_of(d_mm, weight_k), 1),
                "features": {k: round(v, 4) for k, v in feats.items()},
                # --- richer measurements (the "more accurate data") ---
                "circularity": round(circularity(c), 3),      # 1.0 = perfect circle
                "area_px": int(cv2.contourArea(c)),
                "texture_std": round(float(gray[mbool].std()), 1),  # skin texture
                "mean_v": round(float(np.mean(hsv[:, :, 2][mbool])), 1),  # brightness
                "bbox": [int(v) for v in cv2.boundingRect(c)],
                "contour": c,           # used for drawing; removed from JSON
            })

    else:
        onions, coin, px_per_mm, scale_source = [], None, 1.0, "no objects found"
        warnings.append("No onions detected! Check outputs/debug_mask_...jpg - "
                        "onions must be DARKER than the background.")

    # --- counts and percentages ---
    n = len(onions)
    if heap_estimated and n > 1:
        warnings.append(
            f"this looks like a HEAP/pile of onions - the {n} count is an "
            "ESTIMATE (onions hide behind each other). For an exact count, "
            "spread them in ONE layer, or fine-tune YOLO on real photos "
            "(prelabel_real.py -> train_yolo.py).")
    grades = [o["grade"] for o in onions]
    gc = {"A": grades.count("A"), "URS": grades.count("URS"),
          "REJECT": grades.count("REJECT"), "CHECK": grades.count("CHECK")}
    gp = {k: (round(v * 100.0 / n, 1) if n else 0.0) for k, v in gc.items()}
    labels = [o["label"] for o in onions]
    cc = {c: labels.count(c) for c in
          ["GOOD", "DAMAGED", "ROTTEN", "SPROUTED", "UNDERSIZED"]}

    # batch size statistics (mm) - useful summary numbers
    if n:
        ds = [o["diameter_mm"] for o in onions]
        dstats = {"min": round(min(ds), 1), "max": round(max(ds), 1),
                  "mean": round(float(np.mean(ds)), 1),
                  "median": round(float(np.median(ds)), 1),
                  "std": round(float(np.std(ds)), 1)}
    else:
        dstats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    # --- quantity estimate + quality flags + human summary ---
    qflags = quality_checks(gray, hsv)
    if watershed_splits >= 8:
        warnings.append(
            f"heavy splitting ({watershed_splits} cuts) - skin texture, "
            "mold spots or background detail are being separated as "
            "extra 'onions'. If this photo really has only a few onions, "
            "the count is UNRELIABLE - try a plainer background.")
    if n:
        Hf, Wf = bgr.shape[:2]
        edge_ids = [o["id"] for o in onions
                    if o["bbox"][0] <= 2 or o["bbox"][1] <= 2
                    or o["bbox"][0] + o["bbox"][2] >= Wf - 2
                    or o["bbox"][1] + o["bbox"][3] >= Hf - 2]
        if edge_ids:
            qflags.append("onion(s) " + ",".join(map(str, edge_ids)) +
                          " touch the photo edge - count may be incomplete "
                          "(move the camera back)")
        tot_kg = sum(o["mass_g"] for o in onions) / 1000.0
        coverage = round(100.0 * sum(o["area_px"] for o in onions)
                         / (bgr.shape[0] * bgr.shape[1]), 1)
    else:
        tot_kg, coverage = 0.0, 0.0
    summary = build_summary(n, watershed_splits, gc, gp, cc, tot_kg,
                            scale_source, qflags)
    layer_analysis = build_layer_analysis(onions) if n else {
        "layers": [], "note": LAYER_NOTE}

    rep = {
        "batch_id": batch_id or "ONION-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_name,
        "coin_mm": coin_mm,
        "px_per_mm": round(px_per_mm, 3),
        "scale_source": scale_source,
        "onion_count": n,
        "grade_counts": gc,
        "grade_percent": gp,
        "class_counts": cc,
        "diameter_stats": dstats,
        "watershed_splits": watershed_splits,
        "estimated_weight_kg": round(tot_kg, 2),
        "bags_50kg": round(tot_kg / 50.0, 1),
        "weight_k": weight_k,
        "coverage_percent": coverage,
        "quality_flags": qflags,
        "summary": summary,
        "layer_analysis": layer_analysis,
        "onions": onions,   # each onion also keeps its drawing contour here
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
        "settings": {
            "min_area_px": MIN_AREA_PX, "merged_factor": MERGED_FACTOR,
            "circularity_split_max": CIRC_SPLIT_MAX,
            "grade_a_mm": GRADE_A_MM, "urs_mm": URS_MM,
            "thresholds": {"green_sprouted": GREEN_SPROUTED,
                           "dark_rotten": DARK_ROTTEN, "brown_rotten": BROWN_ROTTEN,
                           "dark_damaged": DARK_DAMAGED, "brown_damaged": BROWN_DAMAGED},
        },
    }

    # --- write the 4 report files (skipped in live/in-memory mode) ---
    if out_dir and not in_memory:
        f_ann = os.path.join(out_dir, f"{stem}_annotated.jpg")
        f_jsn = os.path.join(out_dir, f"{stem}_report.json")
        f_txt = os.path.join(out_dir, f"{stem}_report.txt")
        f_card = os.path.join(out_dir, f"{stem}_report_card.jpg")
        f_full = os.path.join(out_dir, f"{stem}_full_report.jpg")
        make_annotated(rep, bgr, coin, f_ann)
        make_report_card(rep, f_ann, f_card)
        make_full_report_jpg(rep, bgr, coin, f_full)
        make_text_report(rep, f_txt)
        with open(f_jsn, "w", encoding="utf-8") as fh:
            # the "contour" key is only for drawing - do not put it in the JSON
            clean = {k: v for k, v in rep.items() if k != "onions"}
            clean["onions"] = [{k: v for k, v in o.items() if k != "contour"}
                               for o in onions]
            # embed a small JPEG of the photo (base64) so the JSON report
            # is self-contained - one file carries data AND the picture
            small = cv2.resize(bgr, (640, int(bgr.shape[0] * 640 / bgr.shape[1])),
                               interpolation=cv2.INTER_AREA)
            ok_enc, buf = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok_enc:
                clean["photo_jpeg_base64"] = base64.b64encode(buf).decode("ascii")
                clean["photo_jpeg_note"] = ("640px JPEG thumbnail embedded "
                                            "(base64); full photo in uploads/")
            json.dump(clean, fh, indent=2)
        rep["files"] = [f_ann, f_jsn, f_txt, f_card, f_full]
    else:
        rep["files"] = []
    return rep


def analyze_frame(bgr, coin_mm=27.0, batch_id="LIVE", distance_mm=None,
                  assume_mm=None, coin_assumed=True):
    """In-memory analysis of ONE live video frame (no files written).
    Used by the web app's live mode and camera.py --auto.
    Note: live frames carry no EXIF, so the distance mode falls back to
    the assumed-size mode automatically (honest, and said so)."""
    return analyze(bgr, coin_mm=coin_mm, batch_id=batch_id, out_dir=None,
                   distance_mm=distance_mm, assume_mm=assume_mm,
                   coin_assumed=coin_assumed)


def print_report(rep):
    """Friendly console summary."""
    print("=" * 64)
    print(" ONION QUALITY REPORT")
    print("=" * 64)
    print(f" image      : {rep['image']}")
    print(f" batch id   : {rep['batch_id']}")
    print(f" date/time  : {rep['timestamp']}")
    print(f" scale      : {rep['px_per_mm']:.2f} px/mm  [{rep['scale_source']}]")
    print(f" onions     : {rep['onion_count']}")
    print(f" est weight : ~ {rep['estimated_weight_kg']} kg "
          f"(~ {rep['bags_50kg']} x 50-kg bags)   [calibratable model]")
    print("-" * 64)
    g, p = rep["grade_counts"], rep["grade_percent"]
    print(f" GRADE A    : {g['A']:3d}  ({p['A']:5.1f}%)")
    print(f" GRADE URS  : {g['URS']:3d}  ({p['URS']:5.1f}%)")
    print(f" REJECT     : {g['REJECT']:3d}  ({p['REJECT']:5.1f}%)")
    if g.get("CHECK"):
        print(f" CHECK      : {g['CHECK']:3d}  ({p.get('CHECK', 0):5.1f}%)"
              "  (partly hidden - size to check)")
    print("-" * 64)
    for o in rep["onions"]:
        f = o["features"]
        print(f"  #{o['id']:<2d} {o['label']:<11s} {o['diameter_mm']:6.1f} mm"
              f"  {o['grade']:<6s} {o.get('layer', '-'):<3s}"
              f"(green {100*f['green']:4.1f}%"
              f"  brown {100*f['brown']:4.1f}%  dark {100*f['dark']:4.1f}%)")
    print("-" * 64)
    if rep["watershed_splits"]:
        print(f" watershed  : {rep['watershed_splits']} touching onion(s) "
              "split into separate onions")
    la = rep.get("layer_analysis") or {}
    for lay in la.get("layers", []):
        print(f" layer {lay['layer']:<3}: {lay['count']:3d} onions"
              f"  (A {lay['grade_percent']['A']}% / URS {lay['grade_percent']['URS']}%"
              f" / REJ {lay['grade_percent']['REJECT']}%"
              f" / CHECK {lay['grade_percent'].get('CHECK', 0)}%)"
              f"  ~{lay['est_weight_kg']} kg"
              f"  avg {lay['avg_diameter_mm']} mm  - {lay['name']}")
    for w in rep["warnings"]:
        print(f" WARNING    : {w}")
    for f in rep.get("quality_flags", []):
        print(f" FLAG       : {f}")
    print(f" SUMMARY    : {rep.get('summary', '')}")
    print(f" NOTE       : {DISCLAIMER}")
    print(f" files      : {', '.join(rep['files'])}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(
        description="Onion Quality Grader - SIH26031 "
                    "(grades VISIBLE SURFACE quality only)")
    ap.add_argument("images", nargs="+", help="path(s) to photo(s) of onion batches")
    ap.add_argument("--coin-mm", type=float, default=27.0,
                    help="reference coin diameter in mm (default 27 = Rs.10/Rs.2)")
    ap.add_argument("--coin", choices=sorted(COIN_MENU),
                    help="coin shortcut: --coin 5 means 23 mm, etc.")
    ap.add_argument("--batch-id", default=None, help="custom batch id")
    ap.add_argument("--out-dir", default="outputs", help="folder for reports")
    ap.add_argument("--distance-mm", type=float, default=None,
                    help="no coin? type the camera distance (e.g. 400 = "
                         "40 cm above the onions) - sizes estimated from "
                         "EXIF optics (+/-20 percent)")
    ap.add_argument("--assume-mm", type=float, default=None,
                    help="no coin? assume a standard onion diameter "
                         "(default 55 mm) - sizes are rough estimates")
    ap.add_argument("--weight-k", type=float, default=WEIGHT_K_G_PER_MM3,
                    help="weight model constant: mass_g = k * diameter_mm^3 "
                         "(default 0.00051 ~ 85 g at 55 mm; calibrate once "
                         "with a scale)")
    args = ap.parse_args()

    coin_mm = COIN_MENU[args.coin] if args.coin else args.coin_mm
    for p in args.images:
        rep = analyze(p, coin_mm=coin_mm, batch_id=args.batch_id,
                      out_dir=args.out_dir, weight_k=args.weight_k,
                      distance_mm=args.distance_mm, assume_mm=args.assume_mm)
        print_report(rep)


if __name__ == "__main__":
    main()
