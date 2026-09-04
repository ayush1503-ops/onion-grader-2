#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytorch_cnn.py - Step 5a: a SMALL custom CNN built from scratch in PyTorch.

It learns to classify ONE onion crop into one of 4 classes:
    good / damaged / rotten / sprouted
(alphabetical order = class indices 0/1/2/3, chosen automatically by
ImageFolder; grader.py uses the same labels. UNDERSIZED is NOT a class -
it stays a size rule in grader.py, measured in mm.)

HONEST LIMIT: this judges VISIBLE SURFACE quality only - a photo cannot
detect internal rot, internal damage or moisture.

Train:      python pytorch_cnn.py
Predict:    python pytorch_cnn.py --predict dataset/test/good/good_dummy_001.jpg

Needs:  dataset/train, dataset/val, dataset/test  -> run
        make_dummy_dataset.py first (or drop real photos in those folders).
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
DATA_DIR    = "dataset"
MODEL_PATH  = "onion_cnn.pt"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3                 # learning rate for Adam
IMG         = 224                  # training image size (224x224)

# ImageNet colour statistics - the standard "centre of the photo world".
# Every pretrained model (Step 6) was trained with them, so we match.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# What ImageFolder gives us when folders are sorted alphabetically.
EXPECTED_CLASSES = ["damaged", "good", "rotten", "sprouted"]

# ----------------------------------------------------------------------------
# transforms: what happens to every image before the model sees it
# ----------------------------------------------------------------------------
# TRAIN: resize + AUGMENT (random changes) so the model sees more variety
# and does not simply memorise the training pictures.
train_tfms = transforms.Compose([
    transforms.Resize((IMG, IMG)),          # tuple = force exact 224x224
    transforms.RandomHorizontalFlip(),      # mirrored onion = same label
    transforms.ColorJitter(brightness=0.2,  # fake slightly brighter/darker
                           contrast=0.2),   # photos without new pictures
    transforms.ToTensor(),                  # PIL image -> tensor in 0..1
    transforms.Normalize(mean=MEAN, std=STD),
])

# VAL / TEST: NO augmentation! We want an honest, repeatable measurement,
# not randomly darkened/flipped images.
eval_tfms = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def make_loaders():
    """ImageFolder: folder name = label. Returns train/val/test loaders."""
    splits = {}
    for split, tfm in (("train", train_tfms), ("val", eval_tfms),
                       ("test", eval_tfms)):
        root = os.path.join(DATA_DIR, split)
        if not os.path.isdir(root):
            raise SystemExit(
                f"Missing {root}.\nCreate the dataset first:\n"
                "    python make_dummy_dataset.py")
        ds = datasets.ImageFolder(root, transform=tfm)
        if ds.classes != EXPECTED_CLASSES:
            raise SystemExit(
                f"{root} has classes {ds.classes}\nexpected {EXPECTED_CLASSES}"
                "\nCheck for extra/missing class folders "
                "(e.g. an old 'cut' folder).")
        splits[split] = DataLoader(ds, batch_size=BATCH_SIZE,
                                   shuffle=(split == "train"))
        print(f"  {split:<5}: {len(ds):3d} images  classes={ds.classes}")
    return splits["train"], splits["val"], splits["test"]


# ----------------------------------------------------------------------------
# the model - exactly 3 conv blocks + 1 classifier head
# ----------------------------------------------------------------------------
def build_model():
    return nn.Sequential(
        # block 1: 3 colours -> 32 filters, 224 -> 112 (MaxPool halves)
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        # block 2: 32 -> 64 filters, 112 -> 56
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        # block 3: 64 -> 128 filters, 56 -> 28
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        # classifier: flatten 128x28x28 numbers -> 128 -> 4 class scores
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 128), nn.ReLU(),
        nn.Dropout(0.5),                     # randomly blank 50% while
        nn.Linear(128, 4),                   # training -> less overfitting
    )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    """Accuracy (0..1) of a model on a DataLoader."""
    model.eval()                             # eval mode: dropout OFF
    correct = total = 0
    with torch.no_grad():                    # no learning bookkeeping
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1)
            correct += (preds == y.to(device)).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train():
    torch.manual_seed(42)                    # reproducible runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset (folder name = label):")
    train_loader, val_loader, test_loader = make_loaders()

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 3 conv blocks + head, {n_params:,} trainable parameters")

    loss_fn = nn.CrossEntropyLoss()          # standard for classification
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining {EPOCHS} epochs "
          f"(batch={BATCH_SIZE}, Adam lr={LR}):\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()                        # train mode: dropout ON
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()                  # 1. forget old gradients
            loss = loss_fn(model(x), y)      # 2. guess + measure error
            loss.backward()                  # 3. compute gradients
            opt.step()                       # 4. nudge weights downhill
            running += loss.item() * y.numel()
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.4f} "
              f"| val acc {val_acc:.4f} | {time.time() - t0:.1f}s")

    # honest final number: the TEST set is touched ONCE, here.
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTEST accuracy (dummy data - plumbing test only): "
          f"{test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    print("\nTry one image:")
    print("  python pytorch_cnn.py --predict "
          "dataset/test/sprouted/sprouted_dummy_003.jpg")


def predict(path):
    """Classify ONE image file and print class + confidences."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No {MODEL_PATH} found - train first: "
                         "python pytorch_cnn.py")

    # class names come from the train folders (alphabetical)
    root = os.path.join(DATA_DIR, "train")
    classes = (sorted(os.listdir(root)) if os.path.isdir(root)
               else EXPECTED_CLASSES)

    from PIL import Image
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image = Image.open(path).convert("RGB")   # same transform as val/test
    x = eval_tfms(image).unsqueeze(0).to(device)   # add batch dimension
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]  # scores -> probabilities
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
    ap = argparse.ArgumentParser(description="Step 5a - custom PyTorch CNN")
    ap.add_argument("--predict", metavar="IMG.jpg",
                    help="classify ONE image instead of training")
    args = ap.parse_args()
    if args.predict:
        predict(args.predict)
    else:
        train()
