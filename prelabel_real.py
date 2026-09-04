#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prelabel_real.py - turn YOUR real onion photos into a YOLO labeling
workspace, with STARTER boxes drawn by the current detector.

WHY THIS EXISTS
---------------
Real-world detection quality needs a YOLO fine-tuned on REAL labeled
photos (Step 7b). Labeling 200 photos from scratch takes hours; letting
the existing detector pre-draw the boxes and CORRECTING them in
LabelImg / Roboflow is several times faster.

HONEST EXPECTATIONS:
  - single layer of onions on a plain surface -> pre-labels are usually
    close, quick to correct
  - heaps on jute sacks / busy backgrounds    -> expect to redraw many
    boxes (that is fine - those are exactly the photos the detector
    needs to learn from!)

WHAT IT MAKES
    dataset_real/images/<name>.jpg      your photos (copied)
    dataset_real/labels/<name>.txt      YOLO boxes: class xc yc w h
                                        (0=good 1=damaged 2=rotten
                                         3=sprouted, values 0..1)
    dataset_real.yaml                   ready for train_yolo.py after
                                        you split train/val

WORKFLOW
    1. python prelabel_real.py path/to/your/photos
    2. Correct the boxes:
         LabelImg : pip install labelImg && labelImg dataset_real/images
                    (open dir, change save dir to dataset_real/labels,
                     output format: YOLO)
         Roboflow : upload dataset_real/images + labels, fix in browser
    3. Split into train/ and val/ (e.g. 80/20) and point
       dataset_real.yaml at them, then:
         python train_yolo.py --data dataset_real.yaml
       (Google Colab T4 GPU recommended for 50 epochs - see its header)

Run:  python prelabel_real.py real_photos
"""

import argparse
import glob
import os
import shutil

import cv2

import grader

NAMES = ["good", "damaged", "rotten", "sprouted"]   # class ids 0..3
# grader labels -> YOLO class ids. UNDERSIZED is a SIZE rule, not a
# quality class - label those onions 'good' and let the size rule run.
LABEL_TO_ID = {"GOOD": 0, "UNDERSIZED": 0, "DAMAGED": 1,
               "ROTTEN": 2, "SPROUTED": 3}


def yolo_line(cls_id, x, y, w, h, W, H):
    """(x, y, w, h) box in pixels -> normalized YOLO line."""
    xc, yc = (x + w / 2.0) / W, (y + h / 2.0) / H
    return f"{cls_id} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}"


def prelabel(photo_path, img_dir, lbl_dir):
    """Pre-label ONE photo. Returns (onions found, lines written)."""
    bgr = cv2.imread(photo_path)
    if bgr is None:
        return 0, []
    H, W = bgr.shape[:2]
    stem = os.path.splitext(os.path.basename(photo_path))[0]

    # in-memory analysis (no report files) - out_dir=None is supported
    rep = grader.analyze(bgr, coin_mm=27.0, out_dir=None)

    lines = []
    for o in rep["onions"]:
        cls_id = LABEL_TO_ID.get(o["label"])
        if cls_id is None:
            continue
        x, y, w, h = o["bbox"]
        lines.append(yolo_line(cls_id, x, y, w, h, W, H))

    cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), bgr)
    with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return rep["onion_count"], lines


def main():
    ap = argparse.ArgumentParser(
        description="Pre-label real onion photos for YOLO training")
    ap.add_argument("photos", help="folder (or glob) with your .jpg photos")
    ap.add_argument("--out", default="dataset_real",
                    help="output workspace folder")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.photos, "*.jpg"))
                   + glob.glob(os.path.join(args.photos, "*.jpeg"))
                   + glob.glob(os.path.join(args.photos, "*.png")))
    if not paths:
        raise SystemExit(f"No photos found in {args.photos}")

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    total_boxes = 0
    for p in paths:
        n, lines = prelabel(p, img_dir, lbl_dir)
        total_boxes += len(lines)
        print(f"  {os.path.basename(p):<40} {n:>3} onion(s) -> "
              f"{len(lines)} starter box(es)")

    yaml_path = os.path.join(args.out + ".yaml")
    with open(yaml_path, "w") as fh:
        fh.write("# YOLO dataset for YOUR real photos (prelabel_real.py)\n"
                 "# 1. correct the boxes (LabelImg / Roboflow)\n"
                 "# 2. split images+labels into train/ and val/ folders\n"
                 "# 3. update the two paths below, then:\n"
                 "#    python train_yolo.py --data " + yaml_path + "\n"
                 "train: dataset_real/train/images\n"
                 "val: dataset_real/val/images\n"
                 "nc: 4\n"
                 "names: ['good', 'damaged', 'rotten', 'sprouted']\n")

    print(f"\nDONE: {len(paths)} photos, {total_boxes} starter boxes")
    print(f" workspace : {img_dir} + {lbl_dir}")
    print(f" next      : correct the boxes, split train/val, then")
    print(f"             python train_yolo.py --data {yaml_path}")
    print("\nStarter boxes come from the classic detector - on heaps and")
    print("busy backgrounds they will need real correcting. That work IS")
    print("the training data that fixes detection for good.")


if __name__ == "__main__":
    main()
