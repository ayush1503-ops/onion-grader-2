#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam_label.py - detector-ASSISTED labeling: build a REAL training set fast.

THE BOTTLENECK: CNNs need hundreds of LABELED real photos. Hand-drawing
boxes is the boring part. This tool does the boring part for you:

 1. it runs the project's onion detector to PROPOSE boxes on each photo
 2. it shows you each proposed crop, big
 3. you press ONE key:  1=good  2=sprouted  3=rotten  4=cut  0=skip
 4. the crop is saved into dataset/<split>/<class>/ ready for training

Result: labeling ~100 photos becomes minutes of key-pressing, and you
never draw a box. (Name: inspired by segment-anything-style assist
labeling; we use our own CV detector, so no huge model download.)

Run:
    python sam_label.py photos_folder/                    # -> dataset/train
    python sam_label.py photos_folder/ --split test       # -> dataset/test
    python sam_label.py photos_folder/ --no-gui           # just list proposals
Keys in the window: 1/2/3/4 = class · 0 = skip crop · s = save whole photo
as 'good' · q = quit.
"""

import argparse
import csv
import glob
import os
import time

import cv2

import grader

CLASSES = ["good", "sprouted", "rotten", "cut"]
CROP = 96                       # saved crop size = training input size


def propose_boxes(image_path):
    """Detector proposals -> list of (x, y, w, h). Uses the project's
    detect-all-onions engine, so shadowed/low-contrast onions are found."""
    img = cv2.imread(image_path)
    if img is None:
        return [], None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = grader.detect_all_onions(img, gray, [])
    return [cv2.boundingRect(c) for c in contours], img


def save_crop(img, box, out_dir, cls, tag):
    """Cut a box (with margin), resize to CROP, save into class folder."""
    x, y, w, h = box
    mx, my = int(w * .10), int(h * .10)                 # 10% margin
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
    folder = os.path.join(out_dir, cls)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{cls}_{tag}.jpg")
    cv2.imwrite(path, crop)
    return path


def main():
    ap = argparse.ArgumentParser(description="assisted labeling tool")
    ap.add_argument("folder", help="folder with photos to label")
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"],
                    help="which dataset split to fill (default train)")
    ap.add_argument("--no-gui", action="store_true",
                    help="headless: only print detector proposals")
    args = ap.parse_args()

    out_dir = os.path.join("dataset", args.split)
    photos = sorted(sum([glob.glob(os.path.join(args.folder, e))
                         for e in ("*.jpg", "*.jpeg", "*.png")], []))
    if not photos:
        raise SystemExit(f"no photos found in {args.folder}")

    log_rows, tag = [], time.strftime("%Y%m%d-%H%M%S")
    for p in photos:
        boxes, img = propose_boxes(p)
        print(f"{os.path.basename(p)}: {len(boxes)} proposal(s)")
        if img is None or args.no_gui:
            continue

        for i, (x, y, w, h) in enumerate(boxes):
            disp = cv2.resize(img[y:y + h, x:x + w], (360, 360)) \
                if w >= 40 and h >= 40 else None
            view = img.copy()
            cv2.rectangle(view, (x, y), (x + w, y + h), (255, 82, 0), 3)
            cv2.putText(view, f"crop {i + 1}/{len(boxes)}  "
                        "1=good 2=sprouted 3=rotten 4=cut 0=skip",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 40, 220), 2)
            cv2.imshow("proposals", view)
            if disp is not None:
                cv2.imshow("crop", disp)
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in map(ord, "01234"):
                    break
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    _write_log(log_rows, tag)
                    print("quit - progress saved to labels_log.csv")
                    return
            cls = {ord(c): CLASSES[j] for j, c in
                   enumerate("1234")}.get(k)
            cv2.destroyWindow("crop") if disp is not None else None
            if cls:
                path = save_crop(img, (x, y, w, h), out_dir, cls,
                                 f"{tag}-{os.path.basename(p)[:-4]}-{i}")
                log_rows.append([os.path.basename(p), i, cls, path])
                print(f"   -> {cls}")
            else:
                print("   -> skipped")
        cv2.destroyWindow("proposals")

    try:
        cv2.destroyAllWindows()          # no-op in headless/no-GUI setups
    except cv2.error:
        pass
    _write_log(log_rows, tag)
    print(f"done - {len(log_rows)} crops labeled into {out_dir}/ "
          f"(log: labels_log_{tag}.csv)")


def _write_log(rows, tag):
    if rows:
        with open(f"labels_log_{tag}.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([["photo", "box", "class", "saved_to"]]
                                     + rows)


if __name__ == "__main__":
    main()
