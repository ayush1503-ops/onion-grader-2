#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_learning.py - Step 6a: TRANSFER LEARNING with ResNet18 (PyTorch).

WHY TRANSFER LEARNING BEATS A CUSTOM CNN WITH FEW IMAGES
--------------------------------------------------------
The Step 5a CNN has 12.9 MILLION parameters and must learn EVERYTHING
(edges, textures, shapes, backgrounds) from YOUR images. With 200 dummy
images (or even 500 real ones) that is far too little - it forgets and
overfits.

ResNet18 was trained on 1.2 MILLION ImageNet photos. Its convolution
layers already know edges, curves, spots, stripes, textures - general
visual knowledge that also applies to onions. So we:
  1. take that trained network              (the "backbone"),
  2. FREEZE it                              (its weights never change),
  3. throw away its last layer              (it knew 1000 ImageNet classes),
  4. bolt on a tiny new head: Linear(512 -> 4) for our 4 onion classes,
  5. train ONLY that head - just 2,052 numbers to learn.

Result: usually better accuracy, faster training, far less data needed.

WHEN TO USE WHICH
  < ~1000 images total                        -> transfer learning (this file)
  thousands of images + unusual domain + GPU  -> custom CNN (Step 5a)
  mobile phone deployment                     -> MobileNetV2 (Step 6b)

HONEST LIMIT: this judges VISIBLE SURFACE quality only - a photo cannot
detect internal rot, internal damage or moisture.

Train:      python transfer_learning.py
Predict:    python transfer_learning.py --predict dataset/test/good/good_dummy_001.jpg

First run downloads the ImageNet weights (~45 MB) automatically and
caches them in ~/.cache/torch - it happens once, ever.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torchvision import models

# Reuse EXACTLY the same data loading / preprocessing / eval helper as
# Step 5a (same classes, same 224x224, same normalization) so the final
# comparison between 5a and 6a is FAIR.
from pytorch_cnn import (EXPECTED_CLASSES, eval_tfms, evaluate,
                         make_loaders)

EPOCHS = 5
LR = 1e-3
RESNET_PATH = "onion_resnet18.pt"          # our fine-tuned model


# ----------------------------------------------------------------------------
# model: pretrained ResNet18 + frozen backbone + fresh 4-class head
# ----------------------------------------------------------------------------
def build_resnet18(n_classes=4):
    """ResNet18 with ImageNet weights, frozen body, new trainable head."""
    try:
        # new torchvision API (>= 0.13); downloads on first use
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:                 # very old torchvision
        net = models.resnet18(pretrained=True)

    for p in net.parameters():             # 2. FREEZE everything...
        p.requires_grad = False

    net.fc = nn.Linear(512, n_classes)     # 4. ...except the new head
    return net                             # (512 = size of resnet18 features)


def train_mode_but_frozen_bn(model):
    """Put model in train mode, but keep BatchNorm in eval mode.

    Why: BatchNorm layers store running statistics (averages) learned on
    ImageNet. In train mode they would UPDATE from our 200 dummy images
    and slowly overwrite that knowledge. We keep them frozen instead.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def collect_predictions(model, loader, device):
    """Run the model over a DataLoader -> (true labels, predicted labels)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.extend(model(x.to(device)).argmax(dim=1).tolist())
            ys.extend(y.tolist())
    return ys, ps


def print_test_report(ys, ps, classes):
    """Honest test-set numbers: accuracy + confusion matrix + per-class."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    acc = sum(int(a == b) for a, b in zip(ys, ps)) / max(len(ys), 1)
    print(f"\nTEST accuracy: {acc:.4f}   ({sum(int(a == b) for a, b in zip(ys, ps))}/{len(ys)})")
    print("\nConfusion matrix (rows = TRUE class, columns = PREDICTED):")
    print(pd.DataFrame(confusion_matrix(ys, ps),
                       index=[f"true_{c}" for c in classes],
                       columns=[f"pred_{c}" for c in classes]))
    print("\nPer-class precision / recall / F1:")
    print(classification_report(ys, ps, target_names=classes,
                                digits=3, zero_division=0))


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset (folder name = label):")
    train_loader, val_loader, test_loader = make_loaders()

    model = build_resnet18().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 backbone: {total:,} params total, "
          f"only {trainable:,} trainable (the new head)")

    loss_fn = nn.CrossEntropyLoss()
    # train ONLY parameters that require gradients (the head)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    print(f"\nTraining head only, {EPOCHS} epochs (Adam lr={LR}):\n")
    for epoch in range(1, EPOCHS + 1):
        train_mode_but_frozen_bn(model)    # dropout ON, BatchNorm frozen
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.4f} "
              f"| val acc {val_acc:.4f} | {time.time() - t0:.1f}s")

    # ---- honest final numbers on the TEST set (touched once) ----
    ys, ps = collect_predictions(model, test_loader, device)
    print_test_report(ys, ps, EXPECTED_CLASSES)

    torch.save(model.state_dict(), RESNET_PATH)
    print(f"\nModel saved to {RESNET_PATH} "
          f"({os.path.getsize(RESNET_PATH) / 1e6:.1f} MB)")
    print("Compare all models honestly:  python evaluate.py")


# ----------------------------------------------------------------------------
# single-image prediction
# ----------------------------------------------------------------------------
def predict(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(RESNET_PATH):
        raise SystemExit(f"No {RESNET_PATH} - train first: "
                         "python transfer_learning.py")

    classes = EXPECTED_CLASSES
    from PIL import Image
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    model.eval()

    x = eval_tfms(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    best = int(torch.argmax(probs))

    print(f"\nImage: {path}")
    print(f"PREDICTION: {classes[best].upper()} "
          f"(confidence {probs[best] * 100:.1f}%)")
    for i, c in enumerate(classes):
        bar = "#" * int(probs[i] * 40)
        print(f"  {c:<9} {probs[i] * 100:5.1f}%  {bar}")
    print("\nReminder: surface quality only - a photo cannot see internal"
          " rot or moisture.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Step 6a - ResNet18 transfer learning (PyTorch)")
    ap.add_argument("--predict", metavar="IMG.jpg",
                    help="classify ONE image instead of training")
    args = ap.parse_args()
    if args.predict:
        predict(args.predict)
    else:
        train()
