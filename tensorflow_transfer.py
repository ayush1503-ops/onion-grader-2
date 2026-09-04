#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensorflow_transfer.py - Step 6b: TRANSFER LEARNING with MobileNetV2
(TensorFlow / Keras).

Same idea as Step 6a (ResNet18 in PyTorch), but the backbone is
MobileNetV2 - a network designed to be LIGHT: ~3.5M parameters vs
ResNet18's 11.7M, and much faster. That matters because the SIH problem
statement asks for a MOBILE app, and MobileNetV2 runs well on phones
(it can also be exported to TFLite for that - see export_models.py).

What happens here:
  1. load MobileNetV2 pretrained on ImageNet, WITHOUT its old top layer
     (include_top=False -> it outputs a 7x7x1280 feature map),
  2. base.trainable = False -> the backbone NEVER changes,
  3. add a small classifier head on top:
        GlobalAveragePooling2D -> Dense(128, relu) -> Dropout -> Dense(4),
  4. train only the head (5 epochs).

HONEST LIMIT: this judges VISIBLE SURFACE quality only - a photo cannot
detect internal rot, internal damage or moisture.

Train:      python tensorflow_transfer.py
Predict:    python tensorflow_transfer.py --predict dataset/test/good/good_dummy_001.jpg

First run downloads the ImageNet weights (~9 MB) automatically and
caches them in ~/.keras/models - it happens once, ever.
"""

import argparse
import os

import numpy as np

DATA_DIR = "dataset"
IMG = 224
BATCH = 32
EPOCHS = 10      # 10 is plenty once inputs are correctly scaled to -1..1
MODEL_PATH = "onion_mobilenetv2.h5"

# Keras sorts class folders alphabetically - SAME order as PyTorch's
# ImageFolder in Steps 5a/6a, so the two frameworks stay consistent.
CLASSES = ["damaged", "good", "rotten", "sprouted"]


# ----------------------------------------------------------------------------
# data: folder name = label
# ----------------------------------------------------------------------------
def make_datasets():
    for split in ("train", "val", "test"):
        if not os.path.isdir(os.path.join(DATA_DIR, split)):
            raise SystemExit(f"Missing {DATA_DIR}/{split} - run "
                             "make_dummy_dataset.py first (or add real photos)")
    kw = dict(image_size=(IMG, IMG), batch_size=BATCH)
    train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), shuffle=True, seed=42, **kw)
    val = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"), shuffle=False, **kw)
    test = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"), shuffle=False, **kw)  # order FIXED
    assert train.class_names == CLASSES, (
        f"class order mismatch: {train.class_names} != {CLASSES}")
    print(f"  train: {len(train)} batches | val: {len(val)} batches "
          f"| test: {len(test)} batches | classes={train.class_names}")
    return train, val, test


# ----------------------------------------------------------------------------
# model: frozen MobileNetV2 + small trainable head
# ----------------------------------------------------------------------------
def build_model():
    # weights="imagenet" = pretrained on 1.2M photos.
    # include_top=False = drop the 1000-class ImageNet output layer.
    # NOTE: MobileNetV2 expects inputs in [-1, 1], but our dataset gives
    # 0..255. Keras 3 does NOT add this preprocessing for us, so we add
    # Rescaling(1/127.5, offset=-1) - exactly what
    # tf.keras.applications.mobilenet_v2.preprocess_input does.
    base = tf.keras.applications.MobileNetV2(weights="imagenet",
                                             include_top=False,
                                             input_shape=(IMG, IMG, 3))
    base.trainable = False                    # freeze the backbone

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMG, IMG, 3)),
        # 0..255 -> -1..1  (MobileNetV2's expected input range)
        tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0),
        # augmentation layers - active ONLY during fit(), never at predict
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(0.2),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),  # 7x7x1280 -> 1280
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation="softmax"),  # 4 class scores
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def print_test_report(y_true, y_pred):
    """Honest test-set numbers: accuracy + confusion matrix + per-class."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    acc = float((y_true == y_pred).mean())
    print(f"\nTEST accuracy: {acc:.4f}   ({int((y_true == y_pred).sum())}/{len(y_true)})")
    print("\nConfusion matrix (rows = TRUE class, columns = PREDICTED):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred),
                       index=[f"true_{c}" for c in CLASSES],
                       columns=[f"pred_{c}" for c in CLASSES]))
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, target_names=CLASSES,
                                digits=3, zero_division=0))


# ----------------------------------------------------------------------------
# training + honest evaluation
# ----------------------------------------------------------------------------
def train():
    tf.random.set_seed(42)
    train_ds, val_ds, test_ds = make_datasets()

    model = build_model()
    model.summary(print_fn=lambda s: None)        # keep output short
    n_train = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    n_total = sum(int(np.prod(w.shape)) for w in model.weights)
    print(f"MobileNetV2 backbone: {n_total:,} params total, "
          f"only {n_train:,} trainable (the new head)")

    print(f"\nTraining head only, {EPOCHS} epochs (Adam):\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    # ---- honest final numbers on the TEST set (touched once) ----
    # shuffle=False above guarantees labels and predictions line up.
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    print_test_report(y_true, y_pred)

    model.save(MODEL_PATH)      # .h5 = single file, easy to copy around
    print(f"\nModel saved to {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    print("(If Keras prints a 'legacy H5' warning - that is fine, the "
          "file works.)")
    print("Compare all models honestly:  python evaluate.py")


# ----------------------------------------------------------------------------
# single-image prediction
# ----------------------------------------------------------------------------
def predict(path):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No {MODEL_PATH} - train first: "
                         "python tensorflow_transfer.py")
    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(path, target_size=(IMG, IMG))
    x = tf.keras.utils.img_to_array(img)[None, ...]   # add batch dimension
    probs = model.predict(x, verbose=0)[0]
    best = int(probs.argmax())

    print(f"\nImage: {path}")
    print(f"PREDICTION: {CLASSES[best].upper()} "
          f"(confidence {probs[best] * 100:.1f}%)")
    for i, c in enumerate(CLASSES):
        bar = "#" * int(probs[i] * 40)
        print(f"  {c:<9} {probs[i] * 100:5.1f}%  {bar}")
    print("\nReminder: surface quality only - a photo cannot see internal"
          " rot or moisture.")


if __name__ == "__main__":
    # import here so --help works even before TensorFlow finishes loading
    import tensorflow as tf
    ap = argparse.ArgumentParser(
        description="Step 6b - MobileNetV2 transfer learning (Keras)")
    ap.add_argument("--predict", metavar="IMG.jpg",
                    help="classify ONE image instead of training")
    args = ap.parse_args()
    if args.predict:
        predict(args.predict)
    else:
        train()
