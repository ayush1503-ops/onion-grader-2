#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dummy_detection_dataset.py - DUMMY detection dataset for YOLO (Step 7b).

Difference from make_dummy_dataset.py (the CLASSIFICATION dataset):
  - classification: ONE onion per image, folder name = label
  - detection:      SEVERAL onions per SCENE + a label .txt per image
                    listing every onion's bounding box

Because WE draw the onions, we KNOW the exact boxes - so the YOLO label
files are written automatically. No manual labeling needed for a plumbing
test. (For real photos you label with Roboflow / LabelImg / CVAT.)

WHAT IT MAKES
  dataset_yolo/train/images/*.jpg + dataset_yolo/train/labels/*.txt  (48)
  dataset_yolo/val/images/*.jpg   + dataset_yolo/val/labels/*.txt    (10)
  dataset_yolo/demo/demo_1.jpg, demo_2.jpg     (2 demo scenes)
  dataset.yaml                                 (YOLO config, spec format)

YOLO label format - one line per onion, all values divided by image size:
  class_id x_center y_center width height
  class ids: 0=good 1=damaged 2=rotten 3=sprouted (order = dataset.yaml)

HONESTY RULE: these are computer-drawn DUMMIES. A detector trained on them
proves the pipeline works - it says NOTHING about real photos.

SAFETY: refuses to delete any file it did not create (real photos safe).

Run:  python make_dummy_detection_dataset.py
"""

import argparse
import os
import re
import shutil

import cv2
import numpy as np

# reuse the defect artwork from the classification dummy generator so both
# datasets look like the same "world"
from make_dummy_dataset import add_damage, add_rot, add_sprout

SIZE = 640                                       # scene size in pixels
NAMES = ["good", "damaged", "rotten", "sprouted"]  # class ids 0..3
SPLITS = {"train": 48, "val": 10}
N_DEMO = 2
SEED = 42
DUMMY_RE = re.compile(r"^(scene|demo)_\d{1,3}\.(jpg|txt)$")


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def background(rng):
    """Dark grainy 'conveyor belt' background, 640x640."""
    img = np.full((SIZE, SIZE, 3), 55, dtype=np.float32)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_onion_at(img, cx, cy, r, rng):
    """One onion body at a GIVEN position (we need to know its box)."""
    if rng.random() < 0.5:                       # golden variety
        body = (int(rng.integers(30, 55)), int(rng.integers(130, 165)),
                int(rng.integers(180, 210)))     # OpenCV = BGR
    else:                                        # red variety
        body = (int(rng.integers(70, 100)), int(rng.integers(55, 80)),
                int(rng.integers(140, 170)))
    cv2.circle(img, (cx, cy), r, body, -1)
    darker = tuple(int(c * 0.78) for c in body)
    for _ in range(int(rng.integers(5, 9))):     # papery skin streaks
        dx = int(rng.integers(-r // 2, r // 2))
        cv2.line(img, (cx + dx, cy - int(0.9 * r)),
                 (cx + dx + int(rng.integers(-4, 5)), cy + int(0.9 * r)),
                 darker, 1)
    for _ in range(int(rng.integers(15, 31))):   # tiny dark speckles
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.92 * r)
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))),
                   1, darker, -1)
    lighter = tuple(min(255, int(c * 1.35)) for c in body)
    cv2.circle(img, (cx - r // 3, cy - r // 3), max(4, r // 6), lighter, -1)


def make_scene(rng):
    """Draw one scene -> (image, [(class_id, x1, y1, x2, y2), ...])."""
    img = background(rng)
    boxes, placed = [], []                       # placed = (cx, cy, r) so far
    n = int(rng.integers(3, 9))                  # 3-8 onions per scene
    for _ in range(n):
        for _try in range(40):                   # rejection sampling: keep
            r = int(rng.integers(45, 85))        # centers apart (small
            cx = int(rng.integers(r + 15, SIZE - r - 15))   # overlaps ok)
            cy = int(rng.integers(r + 85, SIZE - r - 15))   # sprout headroom
            if all((cx - px) ** 2 + (cy - py) ** 2 >= (0.85 * (r + pr)) ** 2
                   for px, py, pr in placed):
                break
        else:
            continue                             # could not place - skip
        placed.append((cx, cy, r))

        cls = NAMES[int(rng.integers(0, len(NAMES)))]
        draw_onion_at(img, cx, cy, r, rng)
        if cls == "damaged":
            add_damage(img, cx, cy, r, rng)
        elif cls == "rotten":
            add_rot(img, cx, cy, r, rng)
        elif cls == "sprouted":
            add_sprout(img, cx, cy, r, rng)

        pad = 6                                  # small safety margin
        sprout = 70 if cls == "sprouted" else 0  # sprouts grow ABOVE the body
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - sprout - pad)
        x2 = min(SIZE, cx + r + pad)
        y2 = min(SIZE, cy + r + pad)
        boxes.append((NAMES.index(cls), x1, y1, x2, y2))

    img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15),
                  0, 255).astype(np.uint8)       # random scene brightness
    return img, boxes


def yolo_lines(boxes):
    """[(class_id, x1, y1, x2, y2)] -> YOLO text lines (normalized)."""
    out = []
    for cls_id, x1, y1, x2, y2 in boxes:
        xc, yc = (x1 + x2) / 2.0 / SIZE, (y1 + y2) / 2.0 / SIZE
        w, h = (x2 - x1) / SIZE, (y2 - y1) / SIZE
        out.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


# ----------------------------------------------------------------------------
# safety: never delete real labeled data
# ----------------------------------------------------------------------------
def find_foreign_file():
    root = "dataset_yolo"
    if not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not DUMMY_RE.match(f):
                return os.path.join(dirpath, f)
    return None


def write_data_yaml():
    """dataset.yaml in the exact format YOLO expects (spec format)."""
    text = ("# YOLOv8 dataset config - generated by "
            "make_dummy_detection_dataset.py\n"
            "train: dataset_yolo/train/images\n"
            "val: dataset_yolo/val/images\n"
            "nc: 4\n"
            "names: ['good', 'damaged', 'rotten', 'sprouted']\n")
    with open("dataset.yaml", "w") as fh:
        fh.write(text)
    print("wrote dataset.yaml:")
    print(text)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Create the DUMMY detection dataset for YOLO (Step 7b)")
    ap.add_argument("--force", action="store_true",
                    help="delete non-dummy files too (DANGER: real data!)")
    args = ap.parse_args()

    foreign = find_foreign_file()
    if foreign and not args.force:
        raise SystemExit(
            f"STOP: found a file that is not one of our dummies:\n  {foreign}\n"
            "Move real labeled data somewhere safe first, or use --force.")

    rng = np.random.default_rng(SEED)
    if os.path.isdir("dataset_yolo"):
        shutil.rmtree("dataset_yolo")

    for split, n in SPLITS.items():
        img_dir = os.path.join("dataset_yolo", split, "images")
        lbl_dir = os.path.join("dataset_yolo", split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        total_boxes = 0
        for i in range(1, n + 1):
            img, boxes = make_scene(rng)
            cv2.imwrite(os.path.join(img_dir, f"scene_{i:03d}.jpg"), img)
            with open(os.path.join(lbl_dir, f"scene_{i:03d}.txt"), "w") as fh:
                fh.write("\n".join(yolo_lines(boxes)) + "\n")
            total_boxes += len(boxes)
        print(f"  dataset_yolo/{split}: {n} scenes, {total_boxes} onion boxes")

    demo_dir = os.path.join("dataset_yolo", "demo")
    os.makedirs(demo_dir, exist_ok=True)
    for i in range(1, N_DEMO + 1):
        img, boxes = make_scene(rng)
        cv2.imwrite(os.path.join(demo_dir, f"demo_{i}.jpg"), img)
        print(f"  demo scene demo_{i}.jpg: {len(boxes)} onions "
              "(for yolo_mode.py demos)")

    write_data_yaml()
    print("\nThese are SYNTHETIC dummies - a model trained on them only")
    print("proves the plumbing. Real accuracy needs real labeled photos.")
    print("\nNext (Step 7b), CPU plumbing test:")
    print("  python train_yolo.py --epochs 8 --imgsz 320 --batch 8")
    print("Real photos: use Google Colab T4 GPU (see train_yolo.py header).")


if __name__ == "__main__":
    main()
