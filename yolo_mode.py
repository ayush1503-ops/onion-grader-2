#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 yolo_mode.py - YOLOv8 AI DETECTION MODE (the "advanced AI" upgrade)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT IT DOES (simple words):
    Classic CV mode finds onions with thresholds + contours.
    THIS mode uses a NEURAL NETWORK (YOLOv8) to find each onion in
    one shot, then reuses the SAME size + color + grading logic on
    every detection. Detection by AI -> measurement/grading shared.

⚠️ HONEST LIMIT (very important - do not remove):
    The free pretrained model (yolov8n.pt) is trained on the COCO
    dataset, which has NO "onion" class. So it CANNOT detect onions
    out of the box. You must FIRST fine-tune it on your own labeled
    onion photos (train_yolo.py, ideally on Google Colab GPU) and
    save the result to:
        models/onion_yolo.pt
    Until that file exists, this module raises ModelNotTrained and
    the app keeps using Classic CV. We do NOT fake YOLO results.

HOW TO RUN (after training):
    python yolo_mode.py test_images/test_batch_1.jpg --coin-mm 27
"""

import argparse
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np

import grader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CUSTOM_MODEL = os.path.join(MODEL_DIR, "onion_yolo.pt")

# class names your trained model should use (train_yolo.py explains)
ONION_CLASS_NAMES = {
    "onion_good": None,      # None -> let the color rules decide
    "onion_damaged": None,
    "onion_rotten": None,
    "onion_sprouted": "SPROUTED",   # green shoot is very reliable
    "onion": None,
}

TRAIN_HELP = (
    "YOLO model not found: models/onion_yolo.pt\n"
    "The free pretrained YOLOv8 does NOT know onions (COCO has no\n"
    "onion class) - it must be fine-tuned on YOUR labeled photos.\n"
    "How to fix (honest way):\n"
    "  1. Take 150-300+ photos of onions in different light.\n"
    "  2. Label them (boxes) with Roboflow / LabelImg / CVAT\n"
    "     using classes: onion_good, onion_damaged, onion_rotten,\n"
    "     onion_sprouted. Export in 'YOLO' format -> dataset.yaml.\n"
    "  3. Train (laptop CPU is slow - Colab T4 GPU is better):\n"
    "        python train_yolo.py --data dataset.yaml\n"
    "  4. The script saves models/onion_yolo.pt - then YOLO mode works."
)


class ModelNotTrained(Exception):
    """Raised when models/onion_yolo.pt does not exist yet."""


_model_cache = None


def model_ready():
    """True only if a fine-tuned onion model exists."""
    return os.path.exists(CUSTOM_MODEL)


def get_model():
    """Load the fine-tuned model once, then reuse it (fast)."""
    global _model_cache
    if not model_ready():
        raise ModelNotTrained(TRAIN_HELP)
    if _model_cache is None:
        from ultralytics import YOLO          # lazy import = faster app start
        _model_cache = YOLO(CUSTOM_MODEL)
    return _model_cache


def detect(bgr, conf=0.25):
    """Run YOLO on one BGR frame -> list of detections."""
    model = get_model()
    res = model.predict(source=bgr, conf=conf, verbose=False)[0]
    names = res.names
    out = []
    for b in res.boxes:
        cls_name = names.get(int(b.cls[0]), "")
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append({
            "bbox_px": [int(x1), int(y1), max(1, int(x2 - x1)),
                        max(1, int(y2 - y1))],
            "conf": round(float(b.conf[0]), 3),
            "yolo_class": cls_name,
            "label_hint": ONION_CLASS_NAMES.get(cls_name, None),
        })
    return out


def _coin_scale(bgr, hsv, coin_mm, distance_mm=None, assume_mm=None):
    """Find the coin exactly like grader.py does (the ruler).
    No coin? Same honest estimate modes as grader.analyze."""
    _, mask = grader.make_object_mask(bgr)
    blobs = grader.get_blobs(mask)
    warnings = []
    coin, px_per_mm, source = None, 1.0, "no scale"
    if blobs:
        median = float(np.median([cv2.contourArea(c) for c in blobs]))
        coin = grader.find_coin(blobs, hsv, median)
    if coin is not None:
        px_per_mm = coin["d_px"] / coin_mm
        source = (f"auto-detected coin, assumed Rs.10/Rs.2 ({coin_mm:g} mm)")
    else:
        # honest estimate mode (live/frames carry no EXIF, so the
        # camera-distance option is only available in grader.py's
        # file mode - here we assume a standard onion size)
        ds = [2 * cv2.minEnclosingCircle(c)[1] for c in blobs] if blobs else []
        med_d_px = float(np.median(ds)) if ds else 1.0
        used = float(assume_mm) if assume_mm else grader.FALLBACK_ONION_MM
        px_per_mm = med_d_px / used
        source = f"NO COIN - assumed median onion {used:g} mm"
        warnings.append("No coin/reference found - sizes are ESTIMATES, "
                        "not measurements. Put a coin in view for exact mm.")
    return coin, px_per_mm, source, warnings


def _measure_and_grade(bgr, dets, coin_mm, distance_mm=None, assume_mm=None):
    """Apply the SAME size + color + grade logic to YOLO detections."""
    gray, _ = grader.make_object_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    coin, px_per_mm, scale_source, warnings = _coin_scale(
        bgr, hsv, coin_mm, distance_mm=distance_mm, assume_mm=assume_mm)

    onions = []
    for i, d in enumerate(dets, 1):
        x, y, w, h = d["bbox_px"]
        H, W = bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        crop_gray = gray[y:y + h, x:x + w]
        crop_hsv = hsv[y:y + h, x:x + w]

        # build an onion mask INSIDE the box (Otsu again, on the crop)
        _, cmask = grader.make_object_mask(
            cv2.cvtColor(cv2.cvtColor(bgr[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        if int(cmask.sum()) < 50:          # segmentation failed -> use box
            cmask = np.full((h, w), 255, np.uint8)

        feats = grader.onion_features(crop_gray, crop_hsv, cmask)
        d_mm = ((w + h) / 2.0) / px_per_mm     # box average width/height
        label = d["label_hint"] or grader.classify(feats, d_mm)
        onions.append({
            "id": i, "label": label,
            "diameter_mm": round(d_mm, 1),
            "grade": grader.grade_of(label, d_mm),
            "mass_g": round(grader.weight_of(d_mm), 1),
            "confidence": d["conf"],       # YOLO detection confidence
            "yolo_class": d["yolo_class"],
            "features": {k: round(v, 4) for k, v in feats.items()},
            "texture_std": round(float(crop_gray[cmask > 0].std()), 1),
            "bbox": [x, y, w, h],
        })

    n = len(onions)
    grades = [o["grade"] for o in onions]
    gc = {"A": grades.count("A"), "URS": grades.count("URS"),
          "REJECT": grades.count("REJECT"), "CHECK": grades.count("CHECK")}
    gp = {k: (round(v * 100.0 / n, 1) if n else 0.0) for k, v in gc.items()}
    labels = [o["label"] for o in onions]
    cc = {c: labels.count(c) for c in
          ["GOOD", "DAMAGED", "ROTTEN", "SPROUTED", "UNDERSIZED"]}
    if n:
        ds = [o["diameter_mm"] for o in onions]
        dstats = {"min": round(min(ds), 1), "max": round(max(ds), 1),
                  "mean": round(float(np.mean(ds)), 1),
                  "median": round(float(np.median(ds)), 1),
                  "std": round(float(np.std(ds)), 1)}
    else:
        dstats = {"min": 0.0, "max": 0.0, "mean": 0.0,
                  "median": 0.0, "std": 0.0}

    tot_kg = sum(o["mass_g"] for o in onions) / 1000.0 if n else 0.0
    qflags = grader.quality_checks(gray, hsv)

    # rough pile-layer estimate for YOLO mode: use BOX overlap as the
    # occlusion cue (a box largely covered by another box = lower layer)
    for i, o in enumerate(onions):
        vis = 1.0
        ax, ay, aw, ah = o["bbox"]
        for j, p in enumerate(onions):
            if i == j:
                continue
            bx, by, bw, bh = p["bbox"]
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            frac = (ix * iy) / max(1, aw * ah)
            vis = min(vis, 1.0 - frac)
        o["visibility"] = round(vis, 3)
        o["layer"] = grader.layer_of(vis)
    # re-grade now that we know which onions are partly hidden
    for o in onions:
        o["grade"] = grader.grade_of(o["label"], o["diameter_mm"],
                                     full_visible=(o["layer"] == "L1"))

    rep = {
        "batch_id": "YOLO-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": "frame",
        "coin_mm": coin_mm,
        "px_per_mm": round(px_per_mm, 3),
        "scale_source": scale_source,
        "detector": "YOLOv8 (fine-tuned onion model)",
        "onion_count": n,
        "grade_counts": gc,
        "grade_percent": gp,
        "class_counts": cc,
        "diameter_stats": dstats,
        "watershed_splits": 0,     # YOLO separates touching onions by itself
        "estimated_weight_kg": round(tot_kg, 2),
        "bags_50kg": round(tot_kg / 50.0, 1),
        "coverage_percent": None,
        "quality_flags": qflags,
        "layer_analysis": grader.build_layer_analysis(onions) if n else
        {"layers": [], "note": grader.LAYER_NOTE},
        "onions": onions,
        "warnings": warnings,
        "disclaimer": grader.DISCLAIMER,
        "settings": {"mode": "yolo", "coin_mm": coin_mm},
    }
    return rep, coin, bgr


def annotate(rep, bgr, coin):
    """Draw YOLO boxes + labels + strips (no contour lines here)."""
    canvas = bgr.copy()
    jobs = []
    if coin is not None:
        cv2.circle(canvas, coin["center"], int(coin["d_px"] / 2), (255, 255, 0), 2)
        jobs.append((f"COIN {rep['coin_mm']:g} mm",
                     coin["center"][0] - 40,
                     coin["center"][1] - int(coin["d_px"] / 2) - 22,
                     (255, 255, 0)))
    for o in rep["onions"]:
        x, y, w, h = o["bbox"]
        color = grader.CLASS_COLORS[o["label"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        conf_txt = f" {o['confidence']*100:.0f}%" if "confidence" in o else ""
        jobs.append((f"#{o['id']} {o['label']} {o['diameter_mm']:.0f}mm"
                     f"{conf_txt} {o['grade']}", x, y, color))

    jobs.sort(key=lambda j: (j[1], j[2]))
    used = []
    for text, lx, ly, color in jobs:
        lx, ly = max(0, lx), max(0, ly)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        rw, rh = tw + 8, th + base + 6
        rect = [lx, ly, lx + rw, ly + rh]
        hits = lambda a, b: not (a[2] < b[0] or b[2] < a[0]
                                 or a[3] < b[1] or b[3] < a[1])
        tries = 0
        while any(hits(rect, u) for u in used) and tries < 25:
            ly += rh + 4
            rect = [lx, ly, lx + rw, ly + rh]
            tries += 1
        used.append(rect)
        cv2.rectangle(canvas, (lx, ly), (lx + rw, ly + rh), (30, 30, 30), -1)
        cv2.putText(canvas, text, (lx + 4, ly + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    p = rep["grade_percent"]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (30, 90, 40), -1)
    cv2.putText(canvas, f"YOLOv8  onions: {rep['onion_count']}  "
                f"A: {p['A']:.0f}%  URS: {p['URS']:.0f}%  REJ: {p['REJECT']:.0f}%",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                1, cv2.LINE_AA)
    strip_y = canvas.shape[0] - 26
    cv2.rectangle(canvas, (0, strip_y), (canvas.shape[1], canvas.shape[0]),
                  (0, 215, 255), -1)
    cv2.putText(canvas, "Visible surface analysis only - cannot detect "
                "internal rot / damage / moisture",
                (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 60, 160), 1, cv2.LINE_AA)
    return canvas


def analyze(image, coin_mm=27.0, batch_id=None, out_dir="outputs",
            distance_mm=None, assume_mm=None):
    """Full YOLO analysis of a photo (path) or frame (numpy BGR).
    distance_mm/assume_mm are forwarded to the shared no-coin scale
    logic inside _measure_and_grade."""
    in_memory = isinstance(image, np.ndarray)
    bgr = grader._fit_width(cv2.imread(image)) if not in_memory \
        else grader._fit_width(image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")

    dets = detect(bgr)
    rep, coin, bgr = _measure_and_grade(bgr, dets, coin_mm,
                                        distance_mm=distance_mm,
                                        assume_mm=assume_mm)
    if batch_id:
        rep["batch_id"] = batch_id
    if not in_memory:
        rep["image"] = image

    if not in_memory and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image))[0]
        f_ann = os.path.join(out_dir, f"{stem}_yolo_annotated.jpg")
        f_jsn = os.path.join(out_dir, f"{stem}_yolo_report.json")
        f_txt = os.path.join(out_dir, f"{stem}_yolo_report.txt")
        f_card = os.path.join(out_dir, f"{stem}_yolo_report_card.jpg")
        cv2.imwrite(f_ann, annotate(rep, bgr, coin))
        grader.make_report_card(rep, f_ann, f_card)
        grader.make_text_report(rep, f_txt)
        clean = {k: v for k, v in rep.items() if k != "onions"}
        clean["onions"] = [{k: v for k, v in o.items() if k != "contour"}
                           for o in rep["onions"]]
        with open(f_jsn, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
        rep["files"] = [f_ann, f_jsn, f_txt, f_card]
    else:
        rep["files"] = []
    return rep


def analyze_frame(bgr, coin_mm=27.0):
    """Live-mode entry: YOLO on one frame, no files, JSON-safe output."""
    rep = analyze(bgr, coin_mm=coin_mm, out_dir=None)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YOLOv8 onion detector")
    ap.add_argument("image", help="photo path")
    ap.add_argument("--coin-mm", type=float, default=27.0)
    args = ap.parse_args()
    try:
        rep = analyze(args.image, coin_mm=args.coin_mm)
        grader.print_report(rep)
    except ModelNotTrained as exc:
        print(str(exc))
        raise SystemExit(1)
