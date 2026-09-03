#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py - builds a SYNTHETIC (computer-drawn) training dataset.

WHY THIS FILE EXISTS
--------------------
To train a neural network you need hundreds of LABELED photos:
    good / sprouted / rotten / cut onions.
We don't have a real photo dataset yet, so this script DRAWS one.
The whole training pipeline becomes runnable end-to-end TODAY, and later
you swap these synthetic images for real farm photos WITHOUT changing
any other file (keep the same folder layout).

HONESTY RULE (never break):
Numbers measured on synthetic data describe SYNTHETIC data only.
They are demo numbers, NOT real-world accuracy. Real accuracy needs a
real labeled test set.

WHAT IT MAKES (folder: dataset/)
    dataset/train/<class>/*.jpg   70%
    dataset/val/<class>/*.jpg     15%
    dataset/test/<class>/*.jpg    15%
Classes: good, sprouted, rotten, cut

Run:  python make_dataset.py
"""

import os
import random
import shutil

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# settings (small on purpose so a laptop CPU can train in minutes)
# ---------------------------------------------------------------------------
SIZE = 96            # every image becomes 96x96 pixels
PER_CLASS = 200      # images per class  (800 total -> minutes, not hours)
CLASSES = ["good", "sprouted", "rotten", "cut"]
OUT = "dataset"
SEED = 42


def draw_onion(img, cx, cy, r, color, rng):
    """One onion body + faint vertical texture lines."""
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    for k in range(-2, 3):
        cv2.ellipse(img, (cx, cy), (max(2, int(r * 0.75)), r),
                    k * 9, 0, 360, tuple(int(c * 0.82) for c in color),
                    1, cv2.LINE_AA)
    # subtle top tip (the onion "neck")
    cv2.line(img, (cx, cy - r),
             (cx + rng.randint(-3, 3), cy - r - rng.randint(3, 8)),
             tuple(int(c * 0.7) for c in color), 2, cv2.LINE_AA)


def make_image(cls, rng):
    """Draw one 96x96 image for class `cls`."""
    bg = rng.randint(70, 235)                       # varied background
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    if rng.random() < .5:                           # sometimes a table edge
        cv2.rectangle(img, (0, SIZE - rng.randint(10, 25)),
                      (SIZE, SIZE), (int(bg * .6),) * 3, -1)

    cx, cy = rng.randint(30, 66), rng.randint(32, 62)
    r = rng.randint(14, 21)
    base = (rng.randint(190, 225), rng.randint(140, 175),
            rng.randint(45, 85))                    # BGR golden onion

    body = base
    if cls == "sprouted":                           # greenish tinge on skin
        body = (int(base[0] * .8), int(base[1] * .9),
                min(255, int(base[2] * .9) + 45))
    if cls == "rotten":                             # darker, sickly skin
        body = (int(base[0] * .45), int(base[1] * .45), int(base[2] * .5))

    draw_onion(img, cx, cy, r, body, rng)

    if cls == "sprouted":                           # the green shoot
        tip = (cx + rng.randint(-4, 4), cy - r - rng.randint(8, 14))
        cv2.line(img, (cx, cy - r + 2), tip, (60, 180, 90), 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r + 4),
                 (tip[0] + rng.randint(-6, 6), tip[1] + rng.randint(2, 6)),
                 (80, 200, 110), 2, cv2.LINE_AA)

    if cls == "rotten":                             # irregular dark patches
        for _ in range(rng.randint(2, 4)):
            px = cx + rng.randint(-r // 2, r // 2)
            py = cy + rng.randint(-r // 2, r // 2)
            cv2.circle(img, (px, py), rng.randint(3, 6), (28, 32, 38), -1)
            cv2.circle(img, (px + 1, py - 1), 2, (55, 60, 66), -1)

    if cls == "cut":                                # pale cut / damage line
        p1 = (cx - r + 3, cy + rng.randint(-6, 6))
        p2 = (cx + r - 3, cy + rng.randint(-6, 6))
        cv2.line(img, p1, p2, (225, 228, 235), 3, cv2.LINE_AA)
        cv2.line(img, p1, p2, (180, 185, 195), 1, cv2.LINE_AA)

    # lighting variation + a little noise (so it doesn't learn pixel positions)
    if rng.random() < .4:
        grad = np.linspace(rng.randint(40, 90), rng.randint(160, 220),
                           SIZE, dtype=np.uint8)
        shade = np.tile(grad, (SIZE, 1))
        img = (img * (shade[:, :, None] / 255.0)).astype(np.uint8)
    noise = np.random.RandomState(SEED + rng.randrange(10_000)).normal(0, 4, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    rng = random.Random(SEED)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)                          # start clean every run
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            os.makedirs(os.path.join(OUT, split, c), exist_ok=True)

    for c in CLASSES:
        n = {"train": int(PER_CLASS * .7), "val": int(PER_CLASS * .15),
             "test": PER_CLASS - int(PER_CLASS * .7) - int(PER_CLASS * .15)}
        idx = 0
        for split, count in n.items():
            for i in range(count):
                img = make_image(c, rng)
                path = os.path.join(OUT, split, c, f"{c}_{idx:04d}.jpg")
                cv2.imwrite(path, img)
                idx += 1
        print(f"  {c:<9} -> {idx} images")

    total = sum(len(f) for _, _, f in os.walk(OUT))
    print(f"\nDone: {total} synthetic images in '{OUT}/' "
          f"(train 70% / val 15% / test 15%)")
    print("Remember: SYNTHETIC demo data - swap in real photos when you have them.")


if __name__ == "__main__":
    main()
