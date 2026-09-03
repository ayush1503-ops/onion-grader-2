#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 train_yolo.py - fine-tune YOLOv8 on YOUR onion photos
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

THE BIG PICTURE (simple words):
    1. You take 150-300+ photos of onions (all defect types).
    2. You DRAW BOXES around every onion and give each box a class
       name (= "labeling"). Tools: Roboflow (web, easiest),
       LabelImg (desktop), CVAT.
    3. Labeling gives you: one .txt file per photo + dataset.yaml.
    4. This script trains YOLOv8n on that data -> models/onion_yolo.pt
    5. After that, yolo_mode.py and the web app's YOLO toggle work.

LABELING CLASSES (use EXACTLY these names):
    onion_good, onion_damaged, onion_rotten, onion_sprouted

dataset.yaml FORMAT (YOLO "detection" format):
    path: /full/path/to/onion-quality-analyzer/dataset_yolo
    train: images/train
    val: images/val
    names:
      0: onion_good
      1: onion_damaged
      2: onion_rotten
      3: onion_sprouted

FOLDER LAYOUT the YAML expects:
    dataset_yolo/
      images/train/*.jpg   images/val/*.jpg
      labels/train/*.txt   labels/val/*.txt
    (each label txt: "class_id cx cy w h" - normalized 0..1.
     Roboflow's "Export -> YOLO" gives you exactly this zip.)

TRAIN (laptop CPU = very slow, Colab T4 GPU = ~20-40 min):
    python train_yolo.py --data dataset.yaml --epochs 50

ON GOOGLE COLAB (free GPU):
    !git clone <your-project>  (or upload the folder)
    %cd onion-quality-analyzer
    !pip install -q ultralytics
    !python train_yolo.py --data dataset.yaml --epochs 50
    then download models/onion_yolo.pt back to your laptop.

HONESTY: after training, report the TEST-set metrics YOLO prints
(mAP50, precision, recall). Never quote numbers you have not
measured on your own labeled test photos.
"""

import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def check_data(data_yaml):
    """Gentle sanity check before training starts."""
    if not os.path.exists(data_yaml):
        raise SystemExit(f"dataset.yaml not found: {data_yaml}\n"
                         "See the top of this file for the format.")
    print(f"OK: found {data_yaml}")
    print("Reminders:")
    print("  - classes must be: onion_good, onion_damaged, "
          "onion_rotten, onion_sprouted")
    print("  - images AND labels must exist for train AND val folders")
    print("  - 150+ photos total is a sane minimum to start")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8n on onions")
    ap.add_argument("--data", default="dataset.yaml",
                    help="path to dataset.yaml (YOLO format)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="start from this pretrained model (nano = fastest)")
    args = ap.parse_args()

    check_data(args.data)

    from ultralytics import YOLO      # imported here = fast --help
    model = YOLO(args.model)          # downloads yolov8n.pt on first use

    print("\n=== TRAINING STARTS (grab a chai) ===")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    # 'best.pt' = the checkpoint that scored best on the VALIDATION set
    best = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(best):
        # newer ultralytics may use runs/ detect/train2, train3 ...
        runs = sorted([d for d in os.listdir("runs/detect")
                       if d.startswith("train")],
                      key=lambda d: os.path.getmtime(os.path.join("runs/detect", d)))
        best = os.path.join("runs", "detect", runs[-1], "weights", "best.pt")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, "onion_yolo.pt")
    shutil.copy(best, dest)
    print(f"\n✅ Saved best model to: {dest}")
    print("YOLO mode in the web app will now work automatically.")

    print("\n=== QUICK VALIDATION ===")
    metrics = model.val()             # honest numbers on the val split
    print(metrics)
    print("\nHONESTY REMINDER: quote ONLY these measured numbers")
    print("(mAP50 / precision / recall) - never invent accuracy. "
          "And remember: the app still grades VISIBLE SURFACE only.")


if __name__ == "__main__":
    main()
