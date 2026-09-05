#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 selftest_presence.py - check the scikit-learn "is it an onion?" model
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

Runs onion_presence.check() over every image that ships with this repo
and verifies the expected answer:

    ONION      image-search/*.jpg          (real onion photos)
               dataset_presence/positive/* (extra real onion photos)
               test_images/test_batch_1..7 (synthetic onion scenes)
    NOT ONION  dataset_presence/negative/* (tomato / potato / apple /
                                            empty table photos)
               test_images/test_batch_8_not_onions.jpg

HONESTY: these are the SAME photos the model was trained on, so this is
a plumbing/regression test, NOT an accuracy claim. The honest,
grouped-cross-validation number is printed by `python train_presence.py`
and stored in the model file (meta.holdout_accuracy).

RUN:  python selftest_presence.py     (exit code 0 = all checks passed)
"""

import glob
import os
import sys

import cv2

import onion_presence as OP

HERE = os.path.dirname(os.path.abspath(__file__))


def cases():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "image-search", "*.jpg"))):
        out.append((p, True))
    for p in sorted(glob.glob(os.path.join(
            HERE, "dataset_presence", "positive", "*"))):
        out.append((p, True))
    for p in sorted(glob.glob(os.path.join(
            HERE, "test_images", "test_batch_*.jpg"))):
        out.append((p, "not_onions" not in os.path.basename(p)))
    for p in sorted(glob.glob(os.path.join(
            HERE, "dataset_presence", "negative", "*"))):
        out.append((p, False))
    return out


def main():
    info = OP.info()
    print("presence model:", info.get("name"),
          "| trees:", info.get("trees"),
          "| CV accuracy:", info.get("holdout_accuracy"))
    if not OP.model_ready():
        print("\nmodels/onion_presence.json is missing - train it first:")
        print("    python train_presence.py")
        return 1

    todo = cases()
    if not todo:
        print("no test images found")
        return 1

    fails = 0
    for path, want in todo:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] unreadable {path}")
            continue
        v = OP.check(img)
        ok = (v["is_onion"] == want)
        fails += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        got = "ONION" if v["is_onion"] else "NOT FOUND"
        print(f"  {mark}  want={'ONION' if want else 'NOT FOUND':9} "
              f"got={got:9} p={v['p_onion']:.2f}  "
              f"{os.path.basename(path)}")

    print(f"\n{len(todo) - fails}/{len(todo)} checks passed")
    if fails:
        print("Re-train with:  python train_presence.py")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
