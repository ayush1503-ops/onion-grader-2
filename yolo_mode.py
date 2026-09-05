#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 yolo_mode.py - Step 7a: YOLOv8 DETECTION + grading (the "AI upgrade")
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

PIPELINE (one photo -> digital quality report):
    1. YOLOv8 finds every onion in the photo and draws a box around it.
    2. Each boxed onion is classified GOOD / DAMAGED / ROTTEN / SPROUTED
       by one of three classifiers (flag --classifier):
           cnn   crop the box -> Step 5/6 CNN classifier (recommended:
                 ResNet18 from Step 6a, falls back to the Step 5a CNN)
           yolo  use the YOLO class directly (only meaningful once YOLO
                 was fine-tuned with the 4 quality classes, Step 7b)
           rules the classic color rules from grader.py (baseline)
    3. Size in mm comes from grader.py's calibration (a coin in the
       photo, or an honest "assumed size" ESTIMATE without one).
    4. UNDERSIZED stays a SIZE rule (like grader.py), not a class.
    5. Grading reuses grader.py exactly: Grade A (45-65 mm) / URS
       (35-70 mm) / REJECT + batch percentages for the report.

⚠️ HONEST LIMITS (do not remove):
    - Visible surface quality only: a photo cannot detect internal rot,
      damage or moisture.
    - The stock pretrained yolov8n.pt is trained on COCO, which has NO
      onion class - it cannot find onions out of the box. For real
      results fine-tune first (train_yolo.py, Step 7b) -> the best
      weights are saved as models/onion_yolo.pt. The web app uses that
      file and honestly refuses to fake YOLO results without it.

HOW TO RUN (CLI):
    python yolo_mode.py photo.jpg --coin-mm 27 --classifier cnn
    python yolo_mode.py photo.jpg --classifier yolo        # fine-tuned only
    python yolo_mode.py photo.jpg --model yolov8n.pt       # demo: stock COCO
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
CUSTOM_MODEL = os.path.join(MODEL_DIR, "onion_yolo.pt")   # fine-tuned model
STOCK_MODEL = "yolov8n.pt"                                # COCO demo model

# class names of the FINE-TUNED model (train_yolo.py) -> grader labels.
# Legacy 'onion_*' names from older train runs are accepted too.
ONION_CLASS_NAMES = {
    "good": "GOOD", "damaged": "DAMAGED",
    "rotten": "ROTTEN", "sprouted": "SPROUTED",
    "onion_good": "GOOD", "onion_damaged": "DAMAGED",
    "onion_rotten": "ROTTEN", "onion_sprouted": "SPROUTED",
    "onion": None,          # generic box -> let the classifier decide
}

TRAIN_HELP = (
    "YOLO model not found: models/onion_yolo.pt\n"
    "The stock pretrained YOLOv8 does NOT know onions (COCO has no onion\n"
    "class) - it must be fine-tuned on labeled onion photos first.\n"
    "How to fix (honest way):\n"
    "  1. Real photos: take 150-300+ photos in different light and label\n"
    "     them with Roboflow / LabelImg / CVAT using classes:\n"
    "       good, damaged, rotten, sprouted   (export 'YOLO' format)\n"
    "     OR, for a plumbing test right now:\n"
    "       python make_dummy_detection_dataset.py\n"
    "  2. Train (CPU test or Google Colab T4 GPU for real data):\n"
    "       python train_yolo.py --epochs 8 --imgsz 320 --batch 8\n"
    "  3. train_yolo.py saves models/onion_yolo.pt - then YOLO mode works."
)


class ModelNotTrained(Exception):
    """Raised when models/onion_yolo.pt does not exist yet."""


_model_cache = (None, None)      # (path, YOLO model)


def model_ready():
    """True only if a fine-tuned onion model exists."""
    return os.path.exists(CUSTOM_MODEL)


def get_model(path=None):
    """Load a YOLO model once, then reuse it (fast).

    path=None -> the fine-tuned models/onion_yolo.pt (app default) and
    raises ModelNotTrained if it is missing. A path can be given for the
    CLI demo (e.g. the stock 'yolov8n.pt').
    """
    global _model_cache
    if path is None:
        if not model_ready():
            raise ModelNotTrained(TRAIN_HELP)
        path = CUSTOM_MODEL
    if _model_cache[0] != path:
        from ultralytics import YOLO          # lazy import = faster start
        _model_cache = (path, YOLO(path))
    return _model_cache[1]


def detect(bgr, conf=0.25, model_path=None):
    """Run YOLO on one BGR frame -> list of detections."""
    model = get_model(model_path)
    res = model.predict(source=bgr, conf=conf, verbose=False)[0]
    names = res.names
    out = []
    for b in res.boxes:
        cls_name = names.get(int(b.cls[0]), "") if isinstance(names, dict) \
            else names[int(b.cls[0])]
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append({
            "bbox_px": [int(x1), int(y1), max(1, int(x2 - x1)),
                        max(1, int(y2 - y1))],
            "conf": round(float(b.conf[0]), 3),
            "yolo_class": cls_name,
            "label_hint": ONION_CLASS_NAMES.get(cls_name, None),
        })
    return out


# ----------------------------------------------------------------------------
# the CNN classifier from Step 5/6 (used when --classifier cnn)
# ----------------------------------------------------------------------------
_cnn_cache = {}


def _cnn_paths():
    """Candidate checkpoints, best first (Step 6a ResNet18 > Step 5a CNN)."""
    return [os.path.join(BASE_DIR, "onion_resnet18.pt"),
            os.path.join(BASE_DIR, "onion_cnn.pt")]


def _get_cnn():
    """Load the best available classifier ONCE. None if never trained."""
    if _cnn_cache:
        return _cnn_cache
    import torch
    from pytorch_cnn import EXPECTED_CLASSES, build_model, eval_tfms
    for path in _cnn_paths():
        if not os.path.exists(path):
            continue
        if "resnet18" in os.path.basename(path):
            from transfer_learning import build_resnet18
            model = build_resnet18()
            name = "ResNet18 transfer (Step 6a)"
        else:
            model = build_model()
            name = "custom CNN (Step 5a)"
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _cnn_cache.update(model=model, name=name,
                          classes=EXPECTED_CLASSES, eval_tfms=eval_tfms)
        return _cnn_cache
    return None


def classify_crop(bgr_crop):
    """Classify ONE BGR crop -> (LABEL, confidence) or None."""
    cnn = _get_cnn()
    if cnn is None:
        return None
    import torch                              # local import: fast startup
    from PIL import Image
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    x = cnn["eval_tfms"](Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cnn["model"](x), dim=1)[0]
    idx = int(probs.argmax())
    return cnn["classes"][idx].upper(), float(probs[idx])


# ----------------------------------------------------------------------------
# size calibration - exactly like grader.py (coin = the ruler)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# measure + classify + grade every detection
# ----------------------------------------------------------------------------
def _measure_and_grade(bgr, dets, coin_mm, distance_mm=None, assume_mm=None,
                       classifier="cnn", skipped_class=0, skipped_names=()):
    """Apply the SAME size + grade logic to YOLO detections.

    classifier: 'cnn'   crop each box -> Step 5/6 CNN classifier
                'yolo'  trust the fine-tuned YOLO class
                'rules' classic color rules from grader.py
    skipped_class: boxes already dropped for a non-onion YOLO class
                   (e.g. 'person') - reported, never graded.
    """
    gray, _ = grader.make_object_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    coin, px_per_mm, scale_source, warnings = _coin_scale(
        bgr, hsv, coin_mm, distance_mm=distance_mm, assume_mm=assume_mm)

    cnn = _get_cnn() if classifier == "cnn" else None
    if classifier == "cnn" and cnn is None:
        warnings.append("No trained CNN found (onion_resnet18.pt / "
                        "onion_cnn.pt) - falling back to YOLO class and "
                        "color rules. Train Step 5a/6a first for CNN labels.")

    # "ONIONS ONLY" also for YOLO boxes: a person-shaped box (tall /
    # deep inside a detected person) is a human, not an onion.
    person_boxes = grader.detect_people(bgr)
    skipped_shape = skipped_person = 0

    def _box_person_frac(bx, by, bw, bh):
        if not person_boxes or bw <= 0 or bh <= 0:
            return 0.0
        hit = 0
        for (px, py, pw, ph) in person_boxes:
            ix = max(0, min(bx + bw, px + pw) - max(bx, px))
            iy = max(0, min(by + bh, py + ph) - max(by, py))
            hit += ix * iy
        return min(1.0, hit / float(bw * bh))

    onions = []
    oid = 0
    for d in dets:
        x, y, w, h = d["bbox_px"]
        H, W = bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 0 or h <= 0:
            continue
        if max(w, h) / max(1, min(w, h)) >= 2.2:
            skipped_shape += 1      # tall/wide person-like box, not an onion
            continue
        if _box_person_frac(x, y, w, h) >= grader.NON_ONION_PERSON_DEEP:
            skipped_person += 1     # box sits on a detected person
            continue
        crop = bgr[y:y + h, x:x + w]
        crop_gray = gray[y:y + h, x:x + w]
        crop_hsv = hsv[y:y + h, x:x + w]

        # build an onion mask INSIDE the box (Otsu again, on the crop)
        _, cmask = grader.make_object_mask(
            cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                         cv2.COLOR_GRAY2BGR))
        if int(cmask.sum()) < 50:          # segmentation failed -> use box
            cmask = np.full((h, w), 255, np.uint8)

        feats = grader.onion_features(crop_gray, crop_hsv, cmask)
        d_mm = ((w + h) / 2.0) / px_per_mm     # box average width/height

        # ---- label: CNN on the crop, else YOLO class, else color rules ----
        label, source, class_conf = None, None, None
        if cnn is not None and w >= 8 and h >= 8:
            r = classify_crop(crop)
            if r is not None:
                label, class_conf = r
                source = "cnn"
        if label is None and d["label_hint"] is not None:
            label, source, class_conf = d["label_hint"], "yolo", d["conf"]
        if label is None:
            label = grader.classify(feats, d_mm)
            source = "rules"

        # UNDERSIZED stays a SIZE rule - exactly like grader.py
        if label == "GOOD" and d_mm < grader.UNDERSIZED_MM:
            label = "UNDERSIZED"

        oid += 1
        onions.append({
            "id": oid, "label": label, "label_source": source,
            "class_conf": (round(class_conf, 3) if class_conf else None),
            "diameter_mm": round(d_mm, 1),
            "grade": grader.grade_of(label, d_mm),
            "mass_g": round(grader.weight_of(d_mm), 1),
            "confidence": d["conf"],       # YOLO box confidence
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

    # "ONIONS ONLY" report: every skipped box is disclosed, never silent
    n_non_onion = skipped_class + skipped_shape + skipped_person
    saw_person = bool(skipped_person) or any(
        str(c).lower() == "person" for c in skipped_names)
    if skipped_class:
        names = ", ".join(sorted(set(map(str, skipped_names)))[:5])
        warnings.append(
            f"{skipped_class} non-onion YOLO box(es) ignored ({names}) - "
            "only onions are graded.")
    if skipped_shape + skipped_person:
        warnings.append(
            f"{skipped_shape + skipped_person} person-shaped YOLO box(es) "
            "ignored (humans are not onions) - keep hands and faces out "
            "of the photo.")

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

    cnn_txt = f" | classifier: {cnn['name']}" if cnn else ""
    rep = {
        "batch_id": "YOLO-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": "frame",
        "coin_mm": coin_mm,
        "px_per_mm": round(px_per_mm, 3),
        "scale_source": scale_source,
        "detector": "YOLOv8 (fine-tuned onion model)",
        "classifier": f"{classifier}{cnn_txt}",
        "onion_count": n,
        "rejected_non_onion": n_non_onion,   # person/non-onion boxes ignored
        "human_detected": bool(saw_person),
        "grade_counts": gc,
        "grade_percent": gp,
        "class_counts": cc,
        "diameter_stats": dstats,
        "watershed_splits": 0,     # YOLO separates touching onions by itself
        "estimated_weight_kg": round(tot_kg, 2),
        "bags_50kg": round(tot_kg / 50.0, 1),
        "weight_k": grader.WEIGHT_K_G_PER_MM3,     # mass model constant
        "summary": grader.build_summary(n, 0, gc, gp, cc, tot_kg,
                                        scale_source, qflags),
        "coverage_percent": None,
        "quality_flags": qflags,
        "layer_analysis": grader.build_layer_analysis(onions) if n else
        {"layers": [], "note": grader.LAYER_NOTE},
        "onions": onions,
        "warnings": warnings,
        "disclaimer": grader.DISCLAIMER,
        "settings": {"mode": "yolo", "coin_mm": coin_mm,
                     "classifier": classifier},
    }
    return rep, coin, bgr


# ----------------------------------------------------------------------------
# drawing (spec: cv2.rectangle + putText for every box + "class conf%")
# ----------------------------------------------------------------------------
def annotate(rep, bgr, coin):
    """Draw YOLO boxes + labels + strips (no contour lines here)."""
    canvas = bgr.copy()
    jobs = []
    if coin is not None:
        cv2.circle(canvas, coin["center"], int(coin["d_px"] / 2),
                   (255, 255, 0), 2)
        jobs.append((f"COIN {rep['coin_mm']:g} mm",
                     coin["center"][0] - 40,
                     coin["center"][1] - int(coin["d_px"] / 2) - 22,
                     (255, 255, 0)))
    for o in rep["onions"]:
        x, y, w, h = o["bbox"]
        color = grader.CLASS_COLORS[o["label"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        # "class conf%" next to every box (spec format)
        conf_txt = ""
        if o.get("class_conf") is not None:
            conf_txt = f" {o['class_conf'] * 100:.0f}%"
        src = o.get("label_source", "")
        jobs.append((f"#{o['id']} {o['label']} {o['diameter_mm']:.0f}mm"
                     f"{conf_txt} {o['grade']}[{src}]",
                     x, y, color))

    jobs.sort(key=lambda j: (j[1], j[2]))
    used = []
    for text, lx, ly, color in jobs:
        lx, ly = max(0, lx), max(0, ly)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.55, 1)
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
                f"A: {p['A']:.0f}%  URS: {p['URS']:.0f}%  "
                f"REJ: {p['REJECT']:.0f}%",
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


# ----------------------------------------------------------------------------
# entry points (the web app uses these two)
# ----------------------------------------------------------------------------
def analyze(image, coin_mm=27.0, batch_id=None, out_dir="outputs",
            distance_mm=None, assume_mm=None, model_path=None,
            classifier="cnn", conf=0.25):
    """Full YOLO analysis of a photo (path) or frame (numpy BGR).

    model_path: None = the fine-tuned models/onion_yolo.pt (raises
    ModelNotTrained when missing - the web app relies on that honesty).
    The CLI can pass the stock 'yolov8n.pt' for a demo instead.
    classifier: 'cnn' | 'yolo' | 'rules' (see module docstring).
    """
    in_memory = isinstance(image, np.ndarray)
    bgr = grader._fit_width(cv2.imread(image)) if not in_memory \
        else grader._fit_width(image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")

    dets = detect(bgr, conf=conf, model_path=model_path)
    # "ONIONS ONLY": keep only known onion classes. A stock COCO model
    # (CLI demo) labels people/chairs/etc. - those must never be graded
    # as onions.
    onion_dets = [d for d in dets if d["yolo_class"] in ONION_CLASS_NAMES]
    skipped_names = [d["yolo_class"] for d in dets
                     if d["yolo_class"] not in ONION_CLASS_NAMES]
    rep, coin, bgr = _measure_and_grade(bgr, onion_dets, coin_mm,
                                        distance_mm=distance_mm,
                                        assume_mm=assume_mm,
                                        classifier=classifier,
                                        skipped_class=len(skipped_names),
                                        skipped_names=skipped_names)
    if model_path:
        rep["detector"] = f"YOLOv8 ({os.path.basename(model_path)})"
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


def analyze_frame(bgr, coin_mm=27.0, classifier="cnn"):
    """Live-mode entry: YOLO on one frame, no files, JSON-safe output."""
    return analyze(bgr, coin_mm=coin_mm, out_dir=None, classifier=classifier)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Step 7a - YOLOv8 onion detection + grading")
    ap.add_argument("image", help="photo path")
    ap.add_argument("--coin-mm", type=float, default=27.0,
                    help="coin diameter for calibration (default Rs.10: 27)")
    ap.add_argument("--classifier", choices=["cnn", "yolo", "rules"],
                    default="cnn",
                    help="who classifies each crop (default: cnn)")
    ap.add_argument("--model", default="auto",
                    help="'auto' = fine-tuned model if present, else stock "
                    "yolov8n.pt (demo only). Or give a .pt path.")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO box confidence threshold")
    args = ap.parse_args()

    if args.model != "auto":
        model_path = args.model
        print(f"Using model: {model_path}")
    elif model_ready():
        model_path = None                     # -> fine-tuned CUSTOM_MODEL
        print(f"Using fine-tuned model: {CUSTOM_MODEL}")
    else:
        model_path = STOCK_MODEL
        print("=" * 68)
        print("WARNING: no fine-tuned model found - using the STOCK")
        print(f"pretrained {STOCK_MODEL} (COCO classes). It has NO onion")
        print("class, so expect few or zero sensible detections.")
        print("Fine-tune first for real results:  python train_yolo.py")
        print("=" * 68)

    try:
        rep = analyze(args.image, coin_mm=args.coin_mm,
                      model_path=model_path, classifier=args.classifier,
                      conf=args.conf)
        grader.print_report(rep)
        if rep["files"]:
            print("\nSaved:")
            for f in rep["files"]:
                print("  " + f)
    except ModelNotTrained as exc:
        print(str(exc))
        raise SystemExit(1)
