#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytorch_cnn.py - a SMALL custom CNN in PyTorch, trained on the synthetic
onion dataset made by make_dataset.py.

This is the "build it from scratch" baseline. On small datasets a
from-scratch CNN usually loses to transfer learning (see
transfer_learning.py) - we train it anyway so you can SEE the difference.
That comparison is the whole lesson.

WHAT IT DOES
 1. loads dataset/train, dataset/val, dataset/test (folder = label)
 2. trains a tiny CNN for a few epochs (CPU-friendly)
 3. picks the best epoch by VALIDATION accuracy
 4. reports TEST accuracy once, at the end (honest: test is touched once)
 5. saves the model to models/pytorch_onion_cnn.pth

Run:  python pytorch_cnn.py
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["good", "sprouted", "rotten", "cut"]
IMG = 96                 # images are resized to 96x96
EPOCHS = 6               # keep small: this is a laptop-CPU demo
LR = 0.001               # learning rate
MODEL_DIR = "models"


# ---------------------------------------------------------------------------
# 1. data
# ---------------------------------------------------------------------------
def load_split(split):
    """Read a dataset folder -> (images float32 N x 3 x 96 x 96, labels)."""
    xs, ys = [], []
    for yi, cname in enumerate(CLASSES):
        folder = os.path.join("dataset", split, cname)
        for fname in sorted(os.listdir(folder)):
            bgr = cv2.imread(os.path.join(folder, fname))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xs.append(rgb)
            ys.append(yi)
    x = np.stack(xs).astype(np.float32) / 255.0     # 0..1
    x = torch.from_numpy(x).permute(0, 3, 1, 2)     # N x 3 x H x W
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# 2. the model (3 conv blocks -> classifier)
# ---------------------------------------------------------------------------
class TinyOnionCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        # 96 -> 48
        x = self.pool(F.relu(self.conv2(x)))        # 48 -> 24
        x = self.pool(F.relu(self.conv3(x)))        # 24 -> 12
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                          # raw scores (logits)


def accuracy(model, x, y, bs=64):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            pred = model(x[i:i + bs]).argmax(1)
            correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(42)

    print("Loading images ...")
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    print(f"train {len(xtr)}  val {len(xva)}  test {len(xte)}")

    model = TinyOnionCNN(len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        total_loss = 0.0
        for i in range(0, len(xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        val_acc = accuracy(model, xva, yva)
        print(f"epoch {epoch}/{EPOCHS}  loss {total_loss/len(xtr):.4f}  "
              f"val_acc {val_acc:.3f}")
        if val_acc > best_val:                      # keep the BEST epoch
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = accuracy(model, xte, yte)
    print(f"\nHONEST RESULT (synthetic data!): "
          f"test accuracy = {test_acc:.3f}  ({len(xte)} images)")

    path = os.path.join(MODEL_DIR, "pytorch_onion_cnn.pth")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, path)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
