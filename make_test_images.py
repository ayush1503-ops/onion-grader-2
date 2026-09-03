#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 make_test_images.py - creates FAKE "photos" of onions for testing
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHY: so you can test grader.py right now, without a camera and
without real onions.

HONESTY: these images are COMPUTER-GENERATED (synthetic), NOT real
onions. They only test the CODE. Real accuracy must be measured on
real photos labeled by a human expert.

WHAT IT CREATES (in the test_images/ folder):
  test_batch_1.jpg          mixed batch + Rs.10-style coin (27 mm)
  test_batch_2_touching.jpg touching pairs + coin
  test_batch_3_no_coin.jpg  no coin (tests the fallback scale warning)

Each onion is drawn at a KNOWN size in mm, so we know the correct
answer ("ground truth") and can check if grader.py counts and
measures correctly.

RUN:  python make_test_images.py
"""

import os
import cv2
import numpy as np

W, H = 1200, 900
BG_V = 245          # light background (onions must be DARKER than this)
PPM = 4.0           # "pixels per millimetre" used while drawing
RNG = np.random.default_rng(42)   # fixed seed -> same images every run


def hsv_bgr(h, s, v):
    """Convert a single HSV color to a BGR tuple OpenCV can draw with."""
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def new_canvas():
    return np.full((H, W, 3), BG_V, np.uint8)


def mm2px(mm):
    """Radius in pixels for a given diameter in mm."""
    return int(mm * PPM / 2.0)


def draw_coin(img, cx, cy, mm, ppm):
    """A gray metal coin. Gray = LOW saturation, so grader can tell it
    apart from colourful onions. The rim is drawn INSIDE the true radius
    so the measured coin diameter stays accurate."""
    r = int(mm / 2 * ppm)
    cv2.circle(img, (cx, cy), r, hsv_bgr(0, 0, 150), -1)          # metal
    cv2.circle(img, (cx, cy), r - 2, hsv_bgr(0, 0, 100), 4)       # rim (inside)
    cv2.circle(img, (cx, cy), int(r * 0.72), hsv_bgr(0, 0, 165), -1)  # face
    cv2.putText(img, "10", (cx - 22, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, hsv_bgr(0, 0, 70), 2, cv2.LINE_AA)


def blotch(img, cx, cy, rr, color, ox=0, oy=0):
    """An irregular blob (bruise / rot patch)."""
    pts = []
    for k in range(14):
        ang = 2 * np.pi * k / 14
        rad = rr * float(RNG.uniform(0.75, 1.2))
        pts.append([int(cx + ox + rad * np.cos(ang)),
                    int(cy + oy + rad * np.sin(ang))])
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def draw_onion(img, cx, cy, r_px, kind):
    """
    Draw one onion.
    kind: 'good' | 'damaged' | 'rotten' | 'sprouted'
    Healthy skin is drawn BRIGHT (V >= 182). Defects are DARKER - that is
    exactly the rule grader.py uses.
    """
    hue = 16 + int(RNG.integers(-3, 4))

    # body with soft radial shading (centre brighter than the rim)
    for i in range(24):
        t = i / 23.0
        r = int(r_px * (1.0 - 0.75 * t))
        v = int(182 + 26 * t)
        cv2.circle(img, (cx, cy), max(1, r), hsv_bgr(hue, 105, v), -1)

    # a slightly darker RIM around the skin (like a real onion edge).
    # Real onions have a visible edge; this gives cv2.watershed a
    # "wall" to cut along when two onions touch. V=168 stays bright
    # enough to NOT count as a defect.
    cv2.circle(img, (cx, cy), r_px - 1, hsv_bgr(hue, 120, 168), 3, cv2.LINE_AA)

    # papery skin texture arcs (still bright -> NOT counted as defects)
    for _ in range(5):
        a = int(RNG.integers(0, 180))
        cv2.ellipse(img, (cx, cy), (int(r_px * 0.85), int(r_px * 0.6)),
                    a, 0, 100, hsv_bgr(hue, 115, 172), 2, cv2.LINE_AA)

    if kind == "damaged":
        # a dark-brown bruise patch (~19% of the surface) + a scratch
        blotch(img, cx, cy, 0.45 * r_px, hsv_bgr(hue + 2, 130, 140),
               ox=int(0.2 * r_px), oy=int(-0.1 * r_px))
        cv2.line(img, (cx - int(0.5 * r_px), cy - int(0.3 * r_px)),
                 (cx + int(0.4 * r_px), cy + int(0.5 * r_px)),
                 hsv_bgr(10, 120, 60), 6, cv2.LINE_AA)

    if kind == "rotten":
        # large very dark blotch (~19% of the surface) + a small one
        blotch(img, cx, cy, 0.45 * r_px, hsv_bgr(12, 120, 45),
               ox=int(-0.15 * r_px), oy=int(0.1 * r_px))
        blotch(img, cx, cy, 0.18 * r_px, hsv_bgr(12, 120, 40),
               ox=int(0.4 * r_px), oy=int(-0.35 * r_px))

    if kind == "sprouted":
        # a green shoot poking out of the top (stays attached to the body)
        cv2.ellipse(img, (cx, cy - int(0.85 * r_px)),
                    (max(3, int(0.16 * r_px)), max(3, int(0.30 * r_px))),
                    0, 0, 360, hsv_bgr(60, 160, 150), -1)


def finish(img, path):
    """Add a little photo-like noise, then save as JPEG.
    Quality 95 keeps edges clean so measurement stays accurate."""
    noise = RNG.normal(0, 4, (H, W, 1))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"created  {path}")


def main():
    os.makedirs("test_images", exist_ok=True)

    # ---------------- batch 1: mixed batch with a coin ----------------
    img = new_canvas()
    draw_coin(img, 80, 830, 27, PPM)
    draw_onion(img, 150, 150, mm2px(55), "good")
    draw_onion(img, 400, 160, mm2px(55), "damaged")
    draw_onion(img, 650, 150, mm2px(55), "rotten")
    draw_onion(img, 900, 210, mm2px(55), "sprouted")
    draw_onion(img, 170, 430, mm2px(30), "good")   # undersized
    draw_onion(img, 480, 480, mm2px(55), "good")   # touching pair - left
    draw_onion(img, 665, 480, mm2px(55), "good")   # touching pair - right
    draw_onion(img, 950, 520, mm2px(40), "good")   # 40 mm -> URS
    finish(img, "test_images/test_batch_1.jpg")
    print("  ground truth: 8 onions | 3 GOOD 55mm (Grade A) | 1 GOOD 40mm (URS)"
          " | 1 DAMAGED | 1 ROTTEN | 1 SPROUTED | 1 UNDERSIZED 30mm | coin 27mm")

    # ---------------- batch 2: touching pairs ----------------
    img = new_canvas()
    draw_coin(img, 1100, 120, 27, PPM)
    draw_onion(img, 300, 300, mm2px(55), "good")
    draw_onion(img, 490, 300, mm2px(55), "good")   # touching pair A
    draw_onion(img, 720, 160, mm2px(55), "sprouted")   # keep 45+ px gap!
    draw_onion(img, 800, 600, mm2px(55), "damaged")
    draw_onion(img, 950, 470, mm2px(55), "rotten")  # touching pair B
    draw_onion(img, 150, 650, mm2px(55), "good")
    finish(img, "test_images/test_batch_2_touching.jpg")
    print("  ground truth: 6 onions | 3 GOOD 55mm (Grade A) | 1 DAMAGED"
          " | 1 ROTTEN | 1 SPROUTED | coin 27mm")

    # ---------------- batch 3: NO coin (fallback scale) ----------------
    img = new_canvas()
    draw_onion(img, 200, 200, mm2px(55), "good")
    draw_onion(img, 500, 200, mm2px(55), "good")
    draw_onion(img, 350, 500, mm2px(55), "good")
    draw_onion(img, 800, 500, mm2px(55), "good")
    finish(img, "test_images/test_batch_3_no_coin.jpg")
    print("  ground truth: 4 GOOD 55mm onions, NO coin -> grader must warn "
          "that sizes are guesses")

    # ---------------- batch 4: DARK background (gunny bag / tray) ------
    # Tests the auto background detection: onions are BRIGHT on a DARK
    # surface, so the mask polarity must flip automatically.
    img = np.full((H, W, 3), 40, np.uint8)          # dark tray ~ V=40
    draw_coin(img, 85, 830, 27, PPM)
    draw_onion(img, 160, 150, mm2px(55), "good")
    draw_onion(img, 430, 160, mm2px(55), "damaged")
    draw_onion(img, 700, 140, mm2px(55), "rotten")
    draw_onion(img, 960, 230, mm2px(55), "sprouted")
    draw_onion(img, 480, 480, mm2px(55), "good")    # touching pair - left
    draw_onion(img, 665, 480, mm2px(55), "good")    # touching pair - right
    finish(img, "test_images/test_batch_4_dark.jpg")
    print("  ground truth: 6 onions on a DARK tray | 2 GOOD 55mm + 1 DAMAGED"
          " + 1 ROTTEN + 1 SPROUTED + 1 touching pair split (2 GOOD) | coin 27mm")

    # ---------------- batch 5: PILE, layer by layer --------------------
    # Front onions are drawn AFTER back onions, so they really COVER
    # them in the image - exactly like a real pile. The grader must:
    #  - split each back+front pair (watershed, 2 seeds)
    #  - see the back onions are mostly hidden -> layer L2
    #  - see the front + single onions fully -> layer L1
    img = new_canvas()
    draw_coin(img, 90, 830, 27, PPM)
    # 4 single onions (keep the median size honest for the splitter)
    draw_onion(img, 160, 150, mm2px(55), "good")
    draw_onion(img, 460, 130, mm2px(55), "good")
    draw_onion(img, 770, 140, mm2px(55), "good")
    draw_onion(img, 1050, 200, mm2px(55), "good")
    # back-row onions (drawn first -> they get covered)
    draw_onion(img, 310, 430, mm2px(55), "good")     # mostly hidden soon
    draw_onion(img, 660, 420, mm2px(55), "damaged")  # mostly hidden soon
    # front-row onions (drawn after -> they cover the backs)
    draw_onion(img, 365, 445, mm2px(55), "good")
    draw_onion(img, 715, 435, mm2px(55), "good")
    finish(img, "test_images/test_batch_5_pile.jpg")
    print("  ground truth: 8 onions | 4 single GOOD + 2 front GOOD covering "
          "2 back onions (1 GOOD, 1 DAMAGED) | layers: 6 on top, 2 hidden")

    print("\nDone! Now run:")
    print("  python grader.py test_images/test_batch_1.jpg --coin-mm 27")
    print("  python grader.py test_images/test_batch_2_touching.jpg --coin-mm 27")
    print("  python grader.py test_images/test_batch_3_no_coin.jpg --coin-mm 27")


if __name__ == "__main__":
    main()
