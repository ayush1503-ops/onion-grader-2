#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - HONEST, side-by-side evaluation of every trained onion
classifier on the TEST SET ONLY.

Models it looks for (skips any that are not trained yet):
    onion_cnn.pt           custom CNN       (Step 5a, pytorch_cnn.py)
    onion_resnet18.pt      ResNet18         (Step 6a, transfer_learning.py)
    onion_mobilenetv2.h5   MobileNetV2      (Step 6b, tensorflow_transfer.py)

Rules of honest evaluation (never break):
  * numbers come from dataset/test ONLY - train/val are never quoted;
  * a model that was never trained is skipped, not invented;
  * if the dataset is the DUMMY one, we say so loudly - dummy numbers
    only prove the plumbing works, nothing about real onions.

Also explains how to read the numbers:
  accuracy    - of all onions, how many got the right label
  precision   - of everything labelled "rotten", how many really were
  recall      - of all truly rotten onions, how many we found
  F1          - one balanced number combining precision + recall
  With 4 balanced classes, accuracy is a fine summary - but per-class
  recall is what matters for grading (missed rot = wrong Grade A %).

Run:  python evaluate.py
"""

import os

import numpy as np

DATA_DIR = "dataset"
CLASSES = ["damaged", "good", "rotten", "sprouted"]
BATCH = 32
IMG = 224


# ----------------------------------------------------------------------------
# shared report printer
# ----------------------------------------------------------------------------
def print_report(name, y_true, y_pred):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    print("=" * 70)
    print(f"MODEL: {name}   |   TEST accuracy {acc:.4f}")
    print("=" * 70)
    print("Confusion matrix (rows = TRUE class, columns = PREDICTED):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred),
                       index=[f"true_{c}" for c in CLASSES],
                       columns=[f"pred_{c}" for c in CLASSES]))
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, target_names=CLASSES,
                                digits=3, zero_division=0))

    # save a picture of the confusion matrix next to the numbers
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        os.makedirs("outputs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=CLASSES, ax=ax,
            xticks_rotation=45, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} (test set)")
        fig.tight_layout()
        out = os.path.join("outputs", f"confusion_{name.split()[0].lower()}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"confusion matrix image -> {out}")
    except Exception as e:                     # plotting is a bonus, not a must
        print(f"(could not save plot: {e})")
    return acc


# ----------------------------------------------------------------------------
# PyTorch models (5a custom CNN, 6a ResNet18)
# ----------------------------------------------------------------------------
def eval_torch(name, path, builder):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from pytorch_cnn import eval_tfms            # same preprocessing as training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = builder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),
                              transform=eval_tfms)
    assert ds.classes == CLASSES, f"unexpected classes {ds.classes}"
    loader = DataLoader(ds, batch_size=BATCH)

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.extend(model(x.to(device)).argmax(dim=1).tolist())
            ys.extend(y.tolist())
    return print_report(name, ys, ps)


# ----------------------------------------------------------------------------
# Keras model (6b MobileNetV2)
# ----------------------------------------------------------------------------
def eval_keras(name, path):
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"), image_size=(IMG, IMG),
        batch_size=BATCH, shuffle=False)         # FIXED order = safe labels
    assert ds.class_names == CLASSES, f"unexpected classes {ds.class_names}"

    model = tf.keras.models.load_model(path)
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    return print_report(name, y_true, y_pred)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    if not os.path.isdir(os.path.join(DATA_DIR, "test")):
        raise SystemExit("No dataset/test - run make_dummy_dataset.py first")

    if os.path.exists(os.path.join(DATA_DIR, "DUMMY_DATA_NOTICE.txt")):
        print("*" * 70)
        print("REMINDER: this is the DUMMY (computer-drawn) dataset.")
        print("Numbers below only prove the pipeline works end-to-end.")
        print("Real accuracy claims need real labeled onion photos.")
        print("*" * 70)

    results = {}

    if os.path.exists("onion_cnn.pt"):
        from pytorch_cnn import build_model
        results["custom CNN (Step 5a)"] = eval_torch(
            "custom CNN (Step 5a)", "onion_cnn.pt", build_model)
    else:
        print("\n[skip] onion_cnn.pt not found - train Step 5a first")

    if os.path.exists("onion_resnet18.pt"):
        from transfer_learning import build_resnet18
        results["ResNet18 (Step 6a)"] = eval_torch(
            "ResNet18 (Step 6a)", "onion_resnet18.pt", build_resnet18)
    else:
        print("[skip] onion_resnet18.pt not found - train Step 6a first")

    if os.path.exists("onion_mobilenetv2.h5"):
        results["MobileNetV2 (Step 6b)"] = eval_keras(
            "MobileNetV2 (Step 6b)", "onion_mobilenetv2.h5")
    else:
        print("[skip] onion_mobilenetv2.h5 not found - train Step 6b first")

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY (test set only)")
        print("=" * 70)
        for name, acc in results.items():
            print(f"  {name:<24} {acc:.4f}")
        print("\nPick the winner on REAL photos, not on dummy circles.")
        print("Reminder: all models judge VISIBLE SURFACE quality only.")


if __name__ == "__main__":
    main()
