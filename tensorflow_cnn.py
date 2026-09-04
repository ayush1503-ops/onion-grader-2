#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensorflow_cnn.py - the SAME idea as pytorch_cnn.py but in TensorFlow/Keras.

Why train the same thing twice? Because SIH26031 asks for BOTH frameworks,
and because seeing the same result in two frameworks proves the PIPELINE
is right, not the framework.

Run:  python tensorflow_cnn.py
"""

import os

import cv2
import numpy as np

CLASSES = ["good", "sprouted", "rotten", "cut"]
IMG = 96
EPOCHS = 6

import tensorflow as tf          # imported late so --help errors stay readable


def load_split(split):
    xs, ys = [], []
    for yi, cname in enumerate(CLASSES):
        folder = os.path.join("dataset", split, cname)
        for fname in sorted(os.listdir(folder)):
            bgr = cv2.imread(os.path.join(folder, fname))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xs.append(rgb)
            ys.append(yi)
    x = np.stack(xs).astype("float32") / 255.0
    y = np.array(ys, dtype="int64")
    return x, y


def build_model():
    """Tiny CNN: 3 conv blocks + dense head (mirrors pytorch_cnn.py)."""
    model = tf.keras.Sequential([
        tf.keras.Input((IMG, IMG, 3)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    os.makedirs("models", exist_ok=True)
    tf.random.set_seed(42)

    print("Loading images ...")
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    print(f"train {len(xtr)}  val {len(xva)}  test {len(xte)}")

    model = build_model()
    model.summary()

    # validation_data drives training; test is NOT touched during training
    model.fit(xtr, ytr, validation_data=(xva, yva),
              epochs=EPOCHS, batch_size=32, verbose=2)

    test_loss, test_acc = model.evaluate(xte, yte, verbose=0)
    print(f"\nHONEST RESULT (synthetic data!): "
          f"test accuracy = {test_acc:.3f}  ({len(xte)} images)")

    path = os.path.join("models", "tensorflow_onion_cnn.keras")
    model.save(path)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
