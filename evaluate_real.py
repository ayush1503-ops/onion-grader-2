#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 evaluate_real.py - HONEST accuracy of the RULES on REAL photos
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT IT DOES (simple words):
    Takes the real labelled photos in image-search/ (fresh, damaged,
    rotten, sprouted), runs the SAME rules the app uses
    (grader.onion_features + grader.classify) on the biggest onion in
    each photo, and prints which ones were right.

HONESTY RULES (never break):
    * these are only ~12 web photos - a SMALL sample. 9/12 does NOT
      mean 75% accuracy on all onions in the world;
    * the labels come from the photo titles/descriptions, checked by
      eye where possible - they are good but not lab-grade;
    * light bruises are known to be invisible to colour rules
      (documented in grader.py) - that line shows as a MISS below
      until a CNN/YOLO is trained on real photos.

RUN:  python evaluate_real.py
=====================================================================
"""

import cv2
import numpy as np

import grader

# ground-truth labels (photo file -> the TRUE class of its main onion).
# Expected app class for a healthy fresh onion is "GOOD".
LABELS = [
    ("FRESH",    "fresh-healthy-onion-whole-bulb-on-plain--1.jpg"),
    ("FRESH",    "fresh-healthy-onion-whole-bulb-on-plain--2.jpg"),
    ("FRESH",    "fresh-healthy-red-onion-whole-bulb-2.jpg"),
    ("DAMAGED",  "damaged-bruised-onion-with-brown-blemish-1.jpg"),
    ("DAMAGED",  "damaged-bruised-onion-with-brown-blemish-2.jpg"),
    ("ROTTEN",   "rotten-spoiled-onion-with-mold-dark-spot-1.jpg"),
    ("ROTTEN",   "rotten-spoiled-onion-with-mold-dark-spot-2.jpg"),
    ("ROTTEN",   "black-moldy-rotten-onion-closeup-1.jpg"),
    ("ROTTEN",   "black-moldy-rotten-onion-closeup-2.jpg"),
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-1.jpg"),
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-2.jpg"),
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-3.jpg"),
]


def main():
    print("Onion grader RULES on REAL labelled photos")
    print("-" * 62)
    correct = 0
    per_class = {}                       # class -> [right, total]
    for truth, fname in LABELS:
        bgr = cv2.imread("image-search/" + fname)
        if bgr is None:
            print(f"[??]  {truth:<9} file missing: {fname}")
            continue
        gray, mask = grader.make_object_mask(bgr)
        blobs = grader.get_blobs(mask)
        if not blobs:
            print(f"[??]  {truth:<9} no onion found:  {fname[:40]}")
            continue
        # the biggest blob = the main onion of the photo
        c = max(blobs, key=cv2.contourArea)
        omask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(omask, [c], -1, 255, -1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        feats = grader.onion_features(gray, hsv, omask)
        # a fresh onion's expected app class is GOOD
        expected = "GOOD" if truth == "FRESH" else truth
        got = grader.classify(feats, 55.0)
        ok = got == expected
        correct += ok
        hits, total = per_class.get(truth, (0, 0))
        per_class[truth] = (hits + ok, total + 1)
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] {truth:<9} -> {got:<9} "
              f"(green_top={feats['green_top']:.2f} "
              f"brown={feats['brown']:.2f} dark={feats['dark']:.2f}) "
              f"{fname[:34]}")
    print("-" * 62)
    for cls, (hits, total) in per_class.items():
        print(f"  {cls:<9} {hits}/{total}")
    print(f"  TOTAL     {correct}/{len(LABELS)}")
    print("\n(small web sample - NOT a scientific accuracy number;" )
    print(" light bruises need a trained model: prelabel_real.py)")


if __name__ == "__main__":
    main()
