#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yolo.py - Step 7b: FINE-TUNE YOLOv8n on your own labeled onion data.

WHY FINE-TUNE
-------------
The stock yolov8n.pt knows the 80 COCO classes (person, car, sports
ball...) but has NO "onion" class, so out of the box it cannot find
onions. Fine-tuning keeps its general detection skills and teaches it our
4 classes: good / damaged / rotten / sprouted.

DATASET FORMAT (dataset.yaml, exact spec format)
------------------------------------------------
    train: dataset_yolo/train/images
    val:   dataset_yolo/val/images
    nc:    4
    names: ['good', 'damaged', 'rotten', 'sprouted']

Every photo needs a twin label file in a "labels" folder (same filename,
.txt, same subpath with "images" replaced by "labels"), one line per
onion, values divided by the image size:
    class_id x_center y_center width height
Example (a good onion in the middle of a 640x640 photo):
    0 0.500000 0.500000 0.250000 0.250000

HOW TO GET LABELED DATA
-----------------------
  real photos : label with Roboflow / LabelImg / CVAT, export "YOLO format"
  dummy test  : python make_dummy_detection_dataset.py
                (draws scenes AND writes the labels - zero manual work)

HOW TO TRAIN
------------
  laptop CPU (slow - fine for the dummy plumbing test only):
      python train_yolo.py --epochs 8 --imgsz 320 --batch 8
  Google Colab (RECOMMENDED for real photos - free T4 GPU):
      1. https://colab.research.google.com  ->  New Notebook
      2. Runtime -> Change runtime type -> T4 GPU
      3. upload your dataset zip (or clone this repo) + train_yolo.py
      4. !pip install ultralytics
      5. !python train_yolo.py                       # 50 epochs, 640px
      6. download runs/detect/train/weights/best.pt
         -> put it here as models/onion_yolo.pt

AFTER TRAINING
--------------
best.pt is copied to models/onion_yolo.pt automatically. Then:
      python yolo_mode.py photo.jpg --classifier cnn

HONEST LIMIT: visible surface quality only - a photo cannot detect
internal rot, damage or moisture.
"""

import argparse
import glob
import os
import shutil

DATA_YAML = "dataset.yaml"
BEST_DEST = os.path.join("models", "onion_yolo.pt")


def ensure_data_yaml(path):
    """Write the default dataset.yaml if missing; print it either way."""
    if os.path.exists(path):
        print(f"Using existing {path}:\n{open(path).read()}")
        return
    text = ("train: dataset_yolo/train/images\n"
            "val: dataset_yolo/val/images\n"
            "nc: 4\n"
            "names: ['good', 'damaged', 'rotten', 'sprouted']\n")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"Wrote default {path}:\n{text}")


def find_best_pt():
    """Newest runs/detect/*/weights/best.pt (fallback if trainer path odd)."""
    cands = sorted(glob.glob(os.path.join("runs", "detect", "*",
                                          "weights", "best.pt")),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser(
        description="Step 7b - fine-tune YOLOv8n on your onion data")
    ap.add_argument("--data", default=DATA_YAML, help="dataset.yaml path")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="starting weights (auto-downloads ~6 MB once); use "
                    "'yolov8n.yaml' to train from scratch with no download")
    ap.add_argument("--scratch", action="store_true",
                    help="force training from scratch (yolov8n.yaml, no "
                    "pretrained weights, fully offline)")
    ap.add_argument("--epochs", type=int, default=50)   # spec default
    ap.add_argument("--imgsz", type=int, default=640)   # spec default
    ap.add_argument("--batch", type=int, default=16)    # spec default
    ap.add_argument("--workers", type=int, default=2,
                    help="data loading workers (low = safer on laptops)")
    args = ap.parse_args()

    ensure_data_yaml(args.data)
    for split in ("train", "val"):
        img_dir = os.path.join("dataset_yolo", split, "images")
        if not os.path.isdir(img_dir):
            raise SystemExit(
                f"Missing {img_dir}.\nCreate the detection dataset first:\n"
                "    python make_dummy_detection_dataset.py\n"
                "(or place your real labeled data in the same layout).")

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("NOTE: no GPU detected - running on CPU.")
        if args.epochs > 10:
            print("  With real photos this is SLOW. For a quick CPU plumbing")
            print("  test use:  --epochs 8 --imgsz 320 --batch 8")
            print("  For real training use Google Colab's free T4 GPU")
            print("  (see the header of this file for the exact steps).")

    # Load the starting model. A .yaml builds the YOLOv8n architecture and
    # trains from scratch (no download at all - works fully offline). A .pt
    # uses pretrained weights; if the download is impossible (offline /
    # blocked network) we fall back to the scratch .yaml so training still
    # runs instead of failing.
    if args.model.endswith(".yaml") or args.scratch:
        model = YOLO("yolov8n.yaml")
        print(f"\nTraining YOLOv8n FROM SCRATCH (no pretrained weights) for "
              f"{args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch})...")
    else:
        try:
            model = YOLO(args.model)          # 'yolov8n.pt' auto-downloads
            print(f"\nFine-tuning {args.model} for {args.epochs} epochs "
                  f"(imgsz={args.imgsz}, batch={args.batch})...")
        except Exception as exc:
            print(f"\nCould not load pretrained weights {args.model}: {exc}")
            print("Falling back to training YOLOv8n FROM SCRATCH "
                  "(yolov8n.yaml) - no download needed. Pretrained COCO "
                  "weights reach higher accuracy faster, but scratch works "
                  "fully offline and still proves the pipeline.")
            model = YOLO("yolov8n.yaml")
            print(f"\nTraining YOLOv8n FROM SCRATCH for {args.epochs} epochs "
                  f"(imgsz={args.imgsz}, batch={args.batch})...")

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers)

    # best.pt = the checkpoint with the best validation mAP
    best = getattr(model.trainer, "best", None)
    if not best or not os.path.exists(str(best)):
        best = find_best_pt()
    if not best:
        raise SystemExit("Training finished but best.pt was not found - "
                         "look inside runs/detect/")

    os.makedirs(os.path.dirname(BEST_DEST), exist_ok=True)
    shutil.copy2(best, BEST_DEST)
    print(f"\nbest.pt copied to {BEST_DEST}")
    print("yolo_mode.py and the web app now use this model automatically.")

    # ---- quick inference check with the trained model: YOLO(best.pt) ----
    print(f"\nInference check with the trained model (YOLO('{best}')):")
    fine = YOLO(str(best))
    val_imgs = sorted(glob.glob(os.path.join("dataset_yolo", "val",
                                             "images", "*.jpg")))
    if val_imgs:
        res = fine.predict(source=val_imgs[0], conf=0.25, verbose=False)[0]
        print(f"  image: {val_imgs[0]}")
        print(f"  onions found: {len(res.boxes)}")
        for b in res.boxes[:6]:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            name = res.names[int(b.cls[0])]
            print(f"    {name:<8} conf {float(b.conf[0]):.2f}   "
                  f"box ({x1},{y1})-({x2},{y2})")
    print("\nFull pipeline demo:")
    print("  python yolo_mode.py dataset_yolo/demo/demo_1.jpg --classifier cnn")
    print("Reminder: surface quality only - photos cannot see internal rot.")


if __name__ == "__main__":
    main()
