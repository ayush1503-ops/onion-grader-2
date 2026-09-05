#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 evaluate_gate.py - HOW ONION-ONLY IS THE "ONION-ONLY" GATE?
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

grader.py contains an APPEARANCE GATE (grader.verify_onions): every
detected blob must look like onion skin before it is counted, measured
and graded. This script MEASURES that gate instead of trusting it.

WHAT YOU NEED (you must supply the photos - nothing is downloaded):

    dataset_gate/onion/       photos that really contain onions
    dataset_gate/not_onion/   photos of things that are NOT onions
                              (apples, potatoes, hands, cups, garlic...)

Every photo is analysed TWICE - once with the gate ON and once with it
OFF - so the numbers show exactly what the gate changed:

    ONIONS KEPT        how many real onions survived the gate
                       (100% = the gate never throws away an onion)
    NON-ONIONS REJECTED how many foreign objects the gate removed
                       (100% = nothing but onions is ever graded)

HONEST EXPECTATION (measured 2026-09 on 14 onion photos + 32
non-onion photos): the gate reliably removes objects whose surface is a
colour onion skin never has - green leaves/shoots, blue plastic, purple
brinjal, mirror-gloss fruit, plain white lids - and it removes a single
foreign object sitting in a pile of real onions. It CANNOT reliably
separate a lone potato / yellow apple / lemon / garlic bulb from an
onion: those are built from the same browns and yellows. Pushing the
thresholds until those are caught starts deleting real onions (measured:
~6% of real onion blobs lost for ~25% of look-alikes caught - we chose
the conservative side). For true onion-only detection train a detector
on labeled photos: prelabel_real.py -> train_yolo.py.

RUN:
    python evaluate_gate.py
    python evaluate_gate.py --onion my_onions --not-onion my_junk
    python evaluate_gate.py --mixed        # also paste each foreign
                                           # object into each onion photo
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

import grader

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def photos(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith(EXT):
            out.append(p)
    return out


def count_with(path, gate_on):
    """Analyse one photo with the gate ON or OFF -> (n_objects, rep)."""
    grader.GATE_ENABLED = gate_on
    try:
        rep = grader.analyze(path, coin_mm=27.0, out_dir=None)
    finally:
        grader.GATE_ENABLED = True
    return rep["onion_count"], rep


# ------------------------------------------------------------------ mixed
def cutout(path):
    """Main foreground object of a (ideally plain-background) photo."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None
    if im.ndim == 3 and im.shape[2] == 4:          # transparent PNG
        m = (im[:, :, 3] > 200).astype(np.uint8) * 255
        bgr = im[:, :, :3]
    else:
        bgr = im
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = (~((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 35))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 2000:
        return None, None
    x, y, w, h = cv2.boundingRect(c)
    return bgr[y:y + h, x:x + w], (m[y:y + h, x:x + w] > 128).astype(np.uint8)


def paste(base, obj, msk, size_px):
    """Paste `obj` into `base` (scaled to size_px). Returns (image, box)."""
    sc = size_px / max(obj.shape[:2])
    obj = cv2.resize(obj, (max(4, int(obj.shape[1] * sc)),
                           max(4, int(obj.shape[0] * sc))),
                     interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (obj.shape[1], obj.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    H, W = base.shape[:2]
    oh, ow = min(obj.shape[0], H - 20), min(obj.shape[1], W - 20)
    obj, msk = obj[:oh, :ow], (msk[:oh, :ow] > 0)
    _g, mk = grader.make_object_mask(base)
    best, rng = None, random.Random(7)
    for _ in range(60):
        x = rng.randint(10, max(11, W - ow - 10))
        y = rng.randint(10, max(11, H - oh - 10))
        fill = float((mk[y:y + oh, x:x + ow] > 0).mean())
        if best is None or fill < best[0]:
            best = (fill, x, y)
    _, x, y = best
    out = base.copy()
    roi = out[y:y + oh, x:x + ow]
    roi[msk] = obj[msk]
    return out, (x, y, ow, oh)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onion", default="dataset_gate/onion")
    ap.add_argument("--not-onion", default="dataset_gate/not_onion")
    ap.add_argument("--mixed", action="store_true",
                    help="also paste every foreign object into every "
                         "onion photo and check it gets removed")
    args = ap.parse_args()

    onion_p, not_p = photos(args.onion), photos(args.not_onion)
    if not onion_p and not not_p:
        sys.exit("No photos found. Create dataset_gate/onion/ and "
                 "dataset_gate/not_onion/ and put photos in them "
                 "(see the docstring at the top of this file).")

    print("ONION-ONLY GATE EVALUATION")
    print("=" * 66)

    kept = total = 0
    if onion_p:
        print(f"\nONION PHOTOS ({len(onion_p)}) - the gate must NOT "
              f"throw real onions away")
        print("-" * 66)
        for p in onion_p:
            off, _ = count_with(p, False)
            on, rep = count_with(p, True)
            kept += on
            total += off
            lost = off - on
            tag = "ok  " if lost == 0 else "LOST"
            print(f"  [{tag}] {os.path.basename(p)[:46]:46s} "
                  f"{on:3d}/{off:3d} onions kept")
            if lost:
                for r in rep.get("not_onion_reasons", [])[:2]:
                    print(f"          reason: {r}")
        if total:
            print(f"  -> ONIONS KEPT: {kept}/{total} "
                  f"({100.0 * kept / total:.1f}%)")

    rej = tot_neg = 0
    if not_p:
        print(f"\nNON-ONION PHOTOS ({len(not_p)}) - the gate should "
              f"remove these objects")
        print("-" * 66)
        for p in not_p:
            off, _ = count_with(p, False)
            on, rep = count_with(p, True)
            rej += max(0, off - on)
            tot_neg += off
            tag = "removed" if on == 0 else ("fewer " if on < off
                                             else "KEPT  ")
            print(f"  [{tag}] {os.path.basename(p)[:46]:46s} "
                  f"{off} -> {on} object(s)")
            if on and rep.get("not_onion_reasons"):
                pass
        if tot_neg:
            print(f"  -> NON-ONIONS REJECTED: {rej}/{tot_neg} "
                  f"({100.0 * rej / tot_neg:.1f}%)")

    if args.mixed and onion_p and not_p:
        print("\nMIXED PHOTOS (one foreign object pasted into an onion "
              "photo)")
        print("-" * 66)
        caught = placed = lost_onion = tot_onion = 0
        for p in onion_p:
            base = grader.read_image(p)
            n0, rep0 = count_with(p, True)
            if not rep0["onions"]:
                continue
            med = float(np.median(
                [2 * cv2.minEnclosingCircle(o["contour"])[1]
                 for o in rep0["onions"]]))
            for q in not_p:
                obj, msk = cutout(q)
                if obj is None:
                    continue
                img, box = paste(base, obj, msk, med)
                tmp = os.path.join("/tmp", "gate_mix.jpg")
                cv2.imwrite(tmp, img)
                n_off, rep_off = count_with(tmp, False)
                n_on, rep_on = count_with(tmp, True)
                off_intr = [o for o in rep_off["onions"]
                            if iou(o["bbox"], box) > 0.3]
                if not off_intr:
                    continue          # the pasted object was not detected
                on_intr = [o for o in rep_on["onions"]
                           if iou(o["bbox"], box) > 0.3]
                placed += 1
                if not on_intr:
                    caught += 1
                # real onions = every other blob that the gate removed
                off_other = [o for o in rep_off["onions"]
                             if iou(o["bbox"], box) <= 0.3]
                on_other = [o for o in rep_on["onions"]
                            if iou(o["bbox"], box) <= 0.3]
                tot_onion += len(off_other)
                lost_onion += max(0, len(off_other) - len(on_other))
        if placed:
            print(f"  -> INTRUDERS REMOVED: {caught}/{placed} "
                  f"({100.0 * caught / placed:.1f}%)")
        if tot_onion:
            print(f"  -> real onions kept in those photos: "
                  f"{tot_onion - lost_onion}/{tot_onion} "
                  f"({100.0 * (tot_onion - lost_onion) / tot_onion:.1f}%)")

    print("\n" + "=" * 66)
    print("Remember: a high 'onions kept' number with a low "
          "'non-onions\nrejected' number is the honest trade-off of an "
          "appearance\ngate. For real onion-only detection, train the "
          "detector:\n  prelabel_real.py -> train_yolo.py")


if __name__ == "__main__":
    main()
