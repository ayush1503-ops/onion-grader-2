#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - HONEST evaluation of every trained model on the TEST set.

Rules this script obeys (our honesty promise):
  * evaluates ONLY on dataset/test (never seen during training)
  * computes REAL numbers and prints them - it never invents or pads them
  * every number is labeled as SYNTHETIC-data result

Outputs:
  * console table + confusion matrix per model
  * outputs/model_evaluation.json  (all numbers, for the report)

Run:  python evaluate.py     (after training the models)
"""

import json
import os

import cv2
import numpy as np

CLASSES = ["good", "sprouted", "rotten", "cut"]
IMG = 96


def confusion(y_true, y_pred, n=len(CLASSES)):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def show(matrix):
    """Pretty-print a confusion matrix."""
    print(" " * 12 + "".join(f"{c[:9]:>10}" for c in CLASSES))
    for i, row in enumerate(matrix):
        print(f"{CLASSES[i][:11]:>12}" + "".join(f"{v:>10}" for v in row))


def load_test():
    xs, ys = [], []
    for yi, cname in enumerate(CLASSES):
        folder = os.path.join("dataset", "test", cname)
        for fname in sorted(os.listdir(folder)):
            xs.append(cv2.imread(os.path.join(folder, fname)))
            ys.append(yi)
    return np.stack(xs), np.array(ys, dtype="int64")


def main():
    os.makedirs("outputs", exist_ok=True)
    x_bgr, y = load_test()
    print(f"TEST set: {len(y)} images (never used in training)\n")

    results = {}

    # ---------------- PyTorch model ----------------
    try:
        import torch
        from pytorch_cnn import TinyOnionCNN
        ckpt = torch.load("models/pytorch_onion_cnn.pth", weights_only=False)
        model = TinyOnionCNN(); model.load_state_dict(ckpt["state_dict"])
        model.eval()
        rgb = x_bgr[:, :, :, ::-1].copy()          # BGR->RGB (numpy, batch-safe)
        x = torch.from_numpy(rgb.astype("float32") / 255.0).permute(0, 3, 1, 2)
        with torch.no_grad():
            pred = model(x).argmax(1).numpy()
        m = confusion(y, pred)
        acc = float((pred == y).mean())
        results["pytorch_custom_cnn"] = {"test_accuracy": acc,
                                         "confusion_matrix": m.tolist()}
        print(f"PyTorch custom CNN   test acc {acc:.3f}  (synthetic data)")
        show(m); print()
    except FileNotFoundError:
        print("PyTorch model not trained yet - skip (run pytorch_cnn.py)\n")

    # ---------------- Keras models ----------------
    import tensorflow as tf
    for name, fname in [("tensorflow_custom_cnn", "tensorflow_onion_cnn.keras"),
                        ("transfer_mobilenetv2", "transfer_mobilenetv2.keras")]:
        path = os.path.join("models", fname)
        if not os.path.exists(path):
            print(f"{name}: not trained yet - skip (run its script)\n")
            continue
        model = tf.keras.models.load_model(path)
        x = x_bgr[:, :, :, ::-1].copy().astype("float32")   # BGR->RGB
        if name == "transfer_mobilenetv2":               # MobileNet wants -1..1
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        else:                                            # plain CNN wants 0..1
            x = x / 255.0
        pred = model.predict(x, verbose=0).argmax(1)
        m = confusion(y, pred)
        acc = float((pred == y).mean())
        results[name] = {"test_accuracy": acc, "confusion_matrix": m.tolist()}
        print(f"{name:<20} test acc {acc:.3f}  (synthetic data)")
        show(m); print()

    with open("outputs/model_evaluation.json", "w") as fh:
        json.dump({"note": "SYNTHETIC demo dataset - not real-world accuracy",
                   "classes": CLASSES, "results": results}, fh, indent=2)
    print("saved -> outputs/model_evaluation.json")


if __name__ == "__main__":
    main()
