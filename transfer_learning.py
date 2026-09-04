#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_learning.py - MobileNetV2 pretrained on ImageNet, re-typed for
onions (TensorFlow/Keras).

WHY TRANSFER LEARNING BEATS A CUSTOM CNN ON <1000 IMAGES
--------------------------------------------------------
A from-scratch CNN must learn EVERYTHING from our 800 pictures: edges,
textures, roundness ... 800 images is far too little for that, so it
overfits (memorises) easily.

MobileNetV2 already learned those low-level skills from 1.2 MILLION
ImageNet photos. We keep that knowledge (freeze the base), and train
only a small classifier head on top. Fewer trainable parameters +
better features = higher accuracy from less data, faster, on CPU.

The trade-offs (be honest about these too):
 + much better accuracy on small data
 + trains in minutes on a laptop CPU
 - the file is bigger (~9 MB vs ~0.3 MB)
 - needs its exact input preprocessing (values -1..1, not 0..1)

Run:  python transfer_learning.py
"""

import os

import cv2
import numpy as np
import tensorflow as tf

CLASSES = ["good", "sprouted", "rotten", "cut"]
IMG = 96
EPOCHS = 4


def load_split(split):
    xs, ys = [], []
    for yi, cname in enumerate(CLASSES):
        folder = os.path.join("dataset", split, cname)
        for fname in sorted(os.listdir(folder)):
            bgr = cv2.imread(os.path.join(folder, fname))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xs.append(rgb)
            ys.append(yi)
    x = np.stack(xs).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # -> -1..1
    y = np.array(ys, dtype="int64")
    return x, y


def main():
    os.makedirs("models", exist_ok=True)
    tf.random.set_seed(42)

    print("Loading images ...")
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    print(f"train {len(xtr)}  val {len(xva)}  test {len(xte)}")

    # 1) the pretrained base - we FREEZE it (trainable=False)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False

    # 2) our small head on top
    model = tf.keras.Sequential([
        tf.keras.Input((IMG, IMG, 3)),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    model.fit(xtr, ytr, validation_data=(xva, yva),
              epochs=EPOCHS, batch_size=32, verbose=2)

    test_loss, test_acc = model.evaluate(xte, yte, verbose=0)
    print(f"\nHONEST RESULT (synthetic data!): "
          f"test accuracy = {test_acc:.3f}  ({len(xte)} images)")

    path = os.path.join("models", "transfer_mobilenetv2.keras")
    model.save(path)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
