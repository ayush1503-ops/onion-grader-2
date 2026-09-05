#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 train_presence.py - train the scikit-learn "IS IT AN ONION?" model
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

Trains a scikit-learn RandomForest that answers ONE question about a
whole photo: does it contain onions, yes or no? The web app calls it
before grading, so a photo of a tomato / a cat / an empty table gets an
honest "Onion not found in this image." instead of fake grades.

TRAINING DATA (all inside this repo - nothing is downloaded)
    ONION (positive)
      - image-search/*.jpg          real onion photos (single + piles)
      - test_images/test_batch_*.jpg synthetic onion scenes
        (test_batch_8_not_onions.jpg is a NEGATIVE, see below)
      - synthetic onion scenes drawn by make_test_images.draw_onion
    NOT ONION (negative)
      - dataset_presence/negative/* real tomato / potato photos
      - test_images/test_batch_8_not_onions.jpg
      - synthetic scenes of make_test_images.draw_not_onion
        (green ball, blue cup, brinjal, glossy apple)
      - plain empty backgrounds (paper, tray, jute-like texture)

Each image is augmented (flips, brightness, blur, rotation) so a few
dozen photos become a few hundred training rows.

OUTPUT
    models/onion_presence.json   - the forest exported as plain numbers
                                   (numpy-only at run time, like
                                   models/onion_clf.json)

RUN
    python train_presence.py
    python train_presence.py --trees 300 --augment 8

HONESTY: the real-photo part of the data is small (a few dozen photos),
so the printed holdout accuracy is a SMALL-SAMPLE number on THIS data,
not a claim about the whole world. Add your own photos to
dataset_presence/positive and dataset_presence/negative to improve it.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import onion_presence as OP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "models", "onion_presence.json")
POS_DIR = os.path.join(HERE, "dataset_presence", "positive")
NEG_DIR = os.path.join(HERE, "dataset_presence", "negative")
RNG = np.random.default_rng(42)

# real onion photos that ship with the repo
REAL_ONIONS = sorted(glob.glob(os.path.join(HERE, "image-search", "*.jpg")))
# synthetic onion scenes (batch 8 is deliberately NOT onions)
SYNTH_ONIONS = [p for p in sorted(glob.glob(os.path.join(
    HERE, "test_images", "test_batch_*.jpg")))
    if "not_onions" not in os.path.basename(p)]
SYNTH_NOT = [p for p in sorted(glob.glob(os.path.join(
    HERE, "test_images", "test_batch_*.jpg")))
    if "not_onions" in os.path.basename(p)]


# ----------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------
def augment(bgr, i):
    """Deterministic-ish variations so few photos -> many rows."""
    img = bgr
    if i % 2 == 1:
        img = cv2.flip(img, 1)
    if i % 3 == 1:
        img = cv2.flip(img, 0)
    if i >= 2:
        f = float(RNG.uniform(0.72, 1.32))
        img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    if i >= 4:
        ang = float(RNG.uniform(-18, 18))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if i >= 6:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.clip(img.astype(np.float32)
                      + RNG.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------
# synthetic scenes (drawn with the repo's own artwork helpers)
# ----------------------------------------------------------------------------
def synth_scenes():
    """Return [(bgr, label, group)] of drawn onion / not-onion scenes."""
    import make_test_images as MT
    out = []
    kinds = ["good", "damaged", "rotten", "sprouted"]
    for s in range(14):                       # onion scenes
        img = MT.new_canvas()
        n = int(RNG.integers(1, 6))
        for j in range(n):
            cx = int(RNG.integers(150, MT.W - 150))
            cy = int(RNG.integers(150, MT.H - 150))
            r = MT.mm2px(float(RNG.uniform(35, 65)))
            MT.draw_onion(img, cx, cy, r, kinds[int(RNG.integers(0, 4))])
        out.append((img, "onion", f"synth_onion_{s}"))

    for s in range(14):                       # not-onion scenes
        img = MT.new_canvas()
        n = int(RNG.integers(1, 5))
        for j in range(n):
            cx = int(RNG.integers(150, MT.W - 150))
            cy = int(RNG.integers(150, MT.H - 150))
            r = MT.mm2px(float(RNG.uniform(35, 65)))
            kind = ["green", "blue", "purple", "glossy"][int(RNG.integers(0, 4))]
            MT.draw_not_onion(img, cx, cy, r, kind)
        out.append((img, "not_onion", f"synth_not_{s}"))

    for s in range(8):                        # EMPTY scenes (no object)
        img = MT.new_canvas()
        if s % 2:                             # add a texture, still empty
            noise = RNG.normal(0, 14, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out.append((img, "not_onion", f"empty_{s}"))
    return out


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
def image_sources():
    """[(path_or_image, label, group)] over every training source."""
    items = []
    for p in REAL_ONIONS:
        items.append((p, "onion", os.path.basename(p)))
    for p in glob.glob(os.path.join(POS_DIR, "*")):
        items.append((p, "onion", os.path.basename(p)))
    for p in SYNTH_ONIONS:
        items.append((p, "onion", os.path.basename(p)))
    # YOLO detection scenes (dark conveyor-belt background) - onions in a
    # very different lighting domain, so the presence model learns that
    # too instead of calling every dark scene "not an onion".
    for p in sorted(glob.glob(os.path.join(
            HERE, "dataset_yolo", "*", "images", "*.jpg")))[::12]:
        items.append((p, "onion", "yolo_" + os.path.basename(p)))
    for p in sorted(glob.glob(os.path.join(
            HERE, "dataset_yolo", "demo", "*.jpg"))):
        items.append((p, "onion", "yolo_" + os.path.basename(p)))
    for p in glob.glob(os.path.join(NEG_DIR, "*")):
        items.append((p, "not_onion", os.path.basename(p)))
    for p in SYNTH_NOT:
        items.append((p, "not_onion", os.path.basename(p)))
    return items


def build(n_aug):
    X, y, groups, seen = [], [], [], []
    for src, label, group in image_sources():
        bgr = cv2.imread(src) if isinstance(src, str) else src
        if bgr is None:
            print(f"  [skip] unreadable: {src}")
            continue
        seen.append((label, group))
        for i in range(n_aug):
            f, _ = OP.features(augment(bgr, i))
            X.append(OP.vec(f))
            y.append(label)
            groups.append(group)
    for bgr, label, group in synth_scenes():
        for i in range(max(2, n_aug // 2)):
            f, _ = OP.features(augment(bgr, i))
            X.append(OP.vec(f))
            y.append(label)
            groups.append(group)
        seen.append((label, group))
    return np.array(X, float), np.array(y), np.array(groups), seen


# ----------------------------------------------------------------------------
# export (plain JSON, numpy-only at run time)
# ----------------------------------------------------------------------------
def export_forest(rf, meta, path):
    trees = []
    for est in rf.estimators_:
        t = est.tree_

        def node(i):
            if t.children_left[i] == -1:
                return [-1, t.value[i][0].tolist()]
            return [int(t.feature[i]), float(t.threshold[i]),
                    node(t.children_left[i]), node(t.children_right[i])]
        trees.append(node(0))
    obj = {"format": 1, "features": OP.FEATURE_ORDER,
           "classes": [str(c) for c in rf.classes_],
           "trees": trees, "meta": meta}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    print(f"exported {len(trees)} trees -> {path} "
          f"({os.path.getsize(path) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(
        description="Train the scikit-learn onion-presence detector")
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--augment", type=int, default=8,
                    help="augmented copies per source photo")
    args = ap.parse_args()

    print("Building the presence dataset (this runs the CV pipeline "
          "on every image)...")
    X, y, groups, seen = build(args.augment)
    n_pos = int((y == "onion").sum())
    print(f"  {len(seen)} source images -> {len(X)} feature rows "
          f"({n_pos} onion / {len(X) - n_pos} not_onion)")
    if len(set(y)) < 2:
        raise SystemExit("Need both onion and not_onion samples.")

    # ---- honest grouped cross-validation (never test on an augmented
    #      copy of a photo the model trained on) ----
    uniq = np.unique(groups)
    folds = min(5, len(uniq))
    gkf = GroupKFold(n_splits=folds)
    accs, wrong = [], []
    for tr, te in gkf.split(X, y, groups):
        m = RandomForestClassifier(n_estimators=args.trees,
                                   max_depth=args.depth,
                                   class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        for g, p, t in zip(groups[te], pred, y[te]):
            if p != t:
                wrong.append(f"{g}: said {p}, truth {t}")
    cv_acc = float(np.mean(accs))
    print(f"\nGrouped {folds}-fold CV accuracy: {cv_acc:.3f} "
          f"(per fold: {', '.join(f'{a:.2f}' for a in accs)})")
    if wrong:
        print("  misclassified (grouped, deduped):")
        for w in sorted(set(wrong))[:12]:
            print("   -", w)

    print("\nTraining the final forest on ALL data...")
    rf = RandomForestClassifier(n_estimators=args.trees,
                                max_depth=args.depth,
                                class_weight="balanced",
                                random_state=0, n_jobs=-1)
    rf.fit(X, y)

    order = np.argsort(rf.feature_importances_)[::-1][:8]
    print("top features: " + ", ".join(
        f"{OP.FEATURE_ORDER[i]}={rf.feature_importances_[i]:.3f}"
        for i in order))

    meta = {
        "model": "sklearn RandomForest (onion presence)",
        "trees": args.trees, "depth": args.depth,
        "samples": int(len(X)), "source_images": int(len(seen)),
        "holdout_accuracy": round(cv_acc, 3),
        "note": ("Grouped cross-validation on THIS repo's photos "
                 "(real onion/tomato/potato photos + synthetic scenes). "
                 "Small sample - not a claim about all photos."),
    }
    export_forest(rf, meta, OUT_PATH)

    # verify the JSON export reproduces sklearn exactly
    OP.load_model(force=True)
    m = OP.load_model()
    same = 0
    for xi, yi in zip(X, y):
        p = OP._proba(m, list(xi))
        pred = m["classes"][int(np.argmax(p))]
        same += int(pred == rf.predict([xi])[0])
    print(f"JSON export matches sklearn on {same}/{len(X)} rows")

    print("\nQuick check on the repo's test images:")
    for p in sorted(glob.glob(os.path.join(HERE, "test_images", "*.jpg"))):
        img = cv2.imread(p)
        v = OP.check(img)
        tag = "ONION" if v["is_onion"] else "NOT FOUND"
        print(f"  {tag:9} p={v['p_onion']:.2f}  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
