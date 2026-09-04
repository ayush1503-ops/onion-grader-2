#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dummy_dataset.py - creates a DUMMY (fake, computer-drawn) dataset so the
whole deep-learning pipeline can be tested END TO END before you have real
onion photos.

WHAT IT MAKES (folder name = label, the format every DL tool understands)
    dataset/train/good/*.jpg       50 images
    dataset/train/damaged/*.jpg    50 images
    dataset/train/rotten/*.jpg     50 images
    dataset/train/sprouted/*.jpg   50 images
    dataset/val/<same 4 classes>/  12 images each
    dataset/test/<same 4 classes>/ 12 images each

Classes match grader.py's labels (good/damaged/rotten/sprouted).
UNDERSIZED is NOT a class here - it stays a SIZE rule in grader.py, because
"too small" is measured in mm, not seen in pixels.

Each image is a random onion-ish circle drawn with OpenCV:
    good     -> clean golden or red onion
    damaged  -> onion + dark cut lines and/or a missing chunk
    rotten   -> onion + dark mold blotches
    sprouted -> onion + green shoots growing out of the top

HONESTY RULE (never break):
    These are NOT photos. Accuracy measured on them only proves the code
    pipeline works ("plumbing test"). It says NOTHING about real-world
    accuracy. When you get real photos, put them in the SAME folders and
    retrain - no other file needs to change.

SAFETY:
    The script refuses to delete anything that is not its own dummy file,
    so your real photos can never be wiped by accident. Use --force to
    override (only if you really mean it).

Run:  python make_dummy_dataset.py
"""

import argparse
import os
import re
import shutil

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
SIZE = 224                                   # same size used for training
CLASSES = ["good", "damaged", "rotten", "sprouted"]
SPLITS = {"train": 50, "val": 12, "test": 12}   # images per class per split
SEED = 42
# our own files always look like: good_dummy_007.jpg  (nothing else matches)
DUMMY_RE = re.compile(r"^(good|damaged|rotten|sprouted)_dummy_\d{3}\.jpg$")


# ----------------------------------------------------------------------------
# drawing helpers (all randomised so the model must learn, not memorise)
# ----------------------------------------------------------------------------
def noisy_background(rng):
    """Dark grainy 'conveyor belt' background."""
    img = np.full((SIZE, SIZE, 3), 55, dtype=np.float32)
    img += rng.normal(0, 8, img.shape)                 # film-grain noise
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_onion(img, rng):
    """Draw the onion body. Returns its center (cx, cy) and radius r."""
    cx = int(rng.integers(78, 147))                    # random position...
    cy = int(rng.integers(88, 150))                    # ...so the model
    r = int(rng.integers(55, 78))                      # ...can't cheat
    if rng.random() < 0.5:                             # golden variety
        body = (int(rng.integers(30, 55)),             # B   (OpenCV = BGR!)
                int(rng.integers(130, 165)),           # G
                int(rng.integers(180, 210)))           # R
    else:                                              # red variety
        body = (int(rng.integers(70, 100)),
                int(rng.integers(55, 80)),
                int(rng.integers(140, 170)))
    cv2.circle(img, (cx, cy), r, body, -1)             # filled circle body
    darker = tuple(int(c * 0.78) for c in body)
    for _ in range(int(rng.integers(5, 9))):           # papery skin streaks
        dx = int(rng.integers(-r // 2, r // 2))
        cv2.line(img, (cx + dx, cy - int(0.9 * r)),
                 (cx + dx + int(rng.integers(-4, 5)), cy + int(0.9 * r)),
                 darker, 1)
    for _ in range(int(rng.integers(15, 31))):         # tiny dark speckles
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.92 * r)
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))),
                   1, darker, -1)
    lighter = tuple(min(255, int(c * 1.35)) for c in body)
    cv2.circle(img, (cx - r // 3, cy - r // 3),        # soft highlight
               max(4, r // 6), lighter, -1)
    return cx, cy, r


def add_damage(img, cx, cy, r, rng):
    """DAMAGED: dark cut lines across the body + sometimes a missing chunk."""
    for _ in range(int(rng.integers(2, 5))):           # 2-4 cut lines
        ang = rng.uniform(0, np.pi)
        half = r * rng.uniform(0.7, 1.15)
        p1 = (int(cx + half * np.cos(ang)), int(cy + half * np.sin(ang)))
        p2 = (int(cx - half * np.cos(ang)), int(cy - half * np.sin(ang)))
        cv2.line(img, p1, p2, (25, 45, 70), int(rng.integers(3, 7)))
    if rng.random() < 0.7:                             # chunk bitten off
        a = rng.uniform(0, 2 * np.pi)
        edge = (int(cx + 0.85 * r * np.cos(a)), int(cy + 0.85 * r * np.sin(a)))
        cv2.circle(img, edge, int(rng.integers(14, 26)), (55, 55, 55), -1)


def add_rot(img, cx, cy, r, rng):
    """ROTTEN: dark brown/black mold blotches INSIDE the body."""
    for _ in range(int(rng.integers(3, 8))):           # 3-7 blotches
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.55 * r)
        p = (int(cx + d * np.cos(a)), int(cy + d * np.sin(a)))
        shade = int(rng.integers(10, 45))
        cv2.circle(img, p, int(rng.integers(8, 20)),
                   (shade // 2, shade, shade + 15), -1)


def add_sprout(img, cx, cy, r, rng):
    """SPROUTED: bright green shoots growing out of the top."""
    for _ in range(int(rng.integers(1, 4))):           # 1-3 shoots
        ang = rng.uniform(-0.5, 0.5)                   # radians off vertical
        length = int(rng.integers(35, 65))
        top = (cx, cy - r)
        tip = (int(cx + length * np.sin(ang)), int(cy - r - length * np.cos(ang)))
        cv2.line(img, top, tip, (60, 190, 60), int(rng.integers(3, 6)))
        cv2.ellipse(img, tip,                          # small leaf at the tip
                    (int(rng.integers(6, 12)), int(rng.integers(3, 6))),
                    float(rng.uniform(0, 180)), 0, 360, (40, 210, 70), -1)


def make_one(cls, rng):
    """Draw one dummy image of class `cls`."""
    img = noisy_background(rng)
    cx, cy, r = draw_onion(img, rng)
    if cls == "damaged":
        add_damage(img, cx, cy, r, rng)
    elif cls == "rotten":
        add_rot(img, cx, cy, r, rng)
    elif cls == "sprouted":
        add_sprout(img, cx, cy, r, rng)                # good = nothing added
    img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15),
                  0, 255).astype(np.uint8)             # random brightness
    return img


# ----------------------------------------------------------------------------
# safety: never delete real photos
# ----------------------------------------------------------------------------
def find_foreign_file():
    """Return path of first file that is NOT one of our dummies, else None."""
    for split in SPLITS:
        root = os.path.join("dataset", split)
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not DUMMY_RE.match(f):
                    return os.path.join(dirpath, f)
    return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create the dummy onion dataset")
    ap.add_argument("--force", action="store_true",
                    help="delete non-dummy files too (DANGER: real photos!)")
    args = ap.parse_args()

    foreign = find_foreign_file()
    if foreign and not args.force:
        raise SystemExit(
            f"STOP: found a file that is NOT a dummy image:\n  {foreign}\n"
            "Move your real photos somewhere safe first, or rerun with "
            "--force if you are sure you want it deleted.")

    rng = np.random.default_rng(SEED)                  # reproducible
    for split in SPLITS:                               # clean rebuild
        root = os.path.join("dataset", split)
        if os.path.isdir(root):
            shutil.rmtree(root)

    total = 0
    for split, n in SPLITS.items():
        for cls in CLASSES:
            folder = os.path.join("dataset", split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(1, n + 1):
                path = os.path.join(folder, f"{cls}_dummy_{i:03d}.jpg")
                cv2.imwrite(path, make_one(cls, rng))
            total += n
            print(f"  {path:<40} {n} images")

    with open(os.path.join("dataset", "DUMMY_DATA_NOTICE.txt"), "w") as fh:
        fh.write("All images here are computer-drawn DUMMIES, not photos.\n"
                 "Accuracy measured on them only proves the code runs.\n"
                 "Replace them with real labeled photos, same folders.\n")

    print(f"\nDONE: {total} dummy images written under dataset/")
    print("These are SYNTHETIC placeholders - do not quote their accuracies")
    print("as real-world performance.")
    print("\nNext step (5a):  python pytorch_cnn.py")


if __name__ == "__main__":
    main()
