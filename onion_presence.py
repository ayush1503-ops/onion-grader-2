#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 onion_presence.py - "IS THERE AN ONION IN THIS PHOTO?" (scikit-learn)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHY THIS FILE EXISTS
--------------------
The CV pipeline (grader.py) and the YOLOv8 engine (yolo_mode.py) both
assume the photo actually holds onions. If a user uploads a cat, a
tomato or an empty table, the honest answer is:

        "Onion not found in this image."

This module is that answer. It is a **scikit-learn Random Forest**
trained by `train_presence.py` on:
    positives - real onion photos (image-search/) + synthetic onion
                scenes drawn by make_test_images / make_dummy_*
    negatives - tomatoes, potatoes, non-onion round objects
                (draw_not_onion), plain / empty backgrounds

RUNTIME IS NUMPY-ONLY
---------------------
The forest is exported to `models/onion_presence.json` exactly like
models/onion_clf.json, so the web app (and Vercel) needs **no
scikit-learn at run time** - only numpy + OpenCV. scikit-learn is
needed to (re-)train.

API
---
    verdict = onion_presence.check(bgr)      # BGR numpy image
    verdict["is_onion"]   -> True / False
    verdict["confidence"] -> 0..1 probability of "onion"
    verdict["reason"]     -> plain-English text for the user
    verdict["model"]      -> "sklearn-rf-json" or "heuristic-fallback"

HONESTY: it decides only whether the picture LOOKS like onions. It
cannot see inside an onion, and a confident "not found" on a weird
photo is still just a model's opinion - the message says so.
"""

import json
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "onion_presence.json")

# feature order - MUST match train_presence.py
FEATURE_ORDER = [
    "cover",         # fraction of the photo taken by object blobs
    "n_blobs",       # how many onion-sized blobs (scaled)
    "circ_med",      # median roundness of the blobs
    "solid_med",     # median solidity
    "area_med",      # median blob area / image area
    "h_med", "h_std",       # hue of the blob interiors
    "s_med", "s_std",       # saturation
    "v_med", "v_std",       # brightness
    "g_std",         # texture (gray std) - papery skin is busy
    "tex_var",       # Laplacian variance (fine skin streaks)
    "green", "cyan_blue", "magenta",   # palette fractions
    "gloss", "neutral_white", "vivid",
    "lab_a", "lab_b",       # onion skin sits in a narrow a/b band
    "red_frac",      # tomato-red / very red pixels
    "edge_den",      # edge density inside the blob
]

NOT_FOUND_MSG = (
    "Onion not found in this image. The AI onion detector "
    "(scikit-learn) did not recognise any onion bulb here. Photograph "
    "whole onions on a plain, contrasting surface with good light."
)

_MODEL = None
_TRIED = False


# ----------------------------------------------------------------------------
# model loading (pure numpy at run time)
# ----------------------------------------------------------------------------
def load_model(force=False):
    """Load models/onion_presence.json once. Returns dict or None."""
    global _MODEL, _TRIED
    if _TRIED and not force:
        return _MODEL
    _TRIED = True
    _MODEL = None
    try:
        with open(MODEL_PATH) as fh:
            m = json.load(fh)
        if m.get("format") == 1 and m.get("features") == FEATURE_ORDER:
            _MODEL = m
    except Exception:
        pass
    return _MODEL


def model_ready():
    return load_model() is not None


def _walk(node, x):
    while node[0] != -1:
        node = node[2] if x[node[0]] <= node[1] else node[3]
    return node[1]


def _proba(model, x):
    """Mean class probability over the forest (same as sklearn)."""
    votes = np.zeros(len(model["classes"]))
    for tree in model["trees"]:
        counts = np.asarray(_walk(tree, x), dtype=float)
        votes += counts / max(1.0, counts.sum())
    return votes / max(1, len(model["trees"]))


# ----------------------------------------------------------------------------
# features (OpenCV) - the SAME code trains and predicts (no mismatch)
# ----------------------------------------------------------------------------
def _blob_mask(bgr):
    """Object blobs via grader's own segmentation (kept identical)."""
    import grader
    gray, mask = grader.make_object_mask(bgr)
    blobs = grader.get_blobs(mask)
    return gray, mask, blobs


def features(bgr):
    """Return (feature dict, n_blobs). Works on any BGR image."""
    import grader
    bgr = grader._fit_width(bgr)
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    gray, mask, blobs = _blob_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    f = {k: 0.0 for k in FEATURE_ORDER}
    f["cover"] = float((mask > 0).mean())
    f["n_blobs"] = min(1.0, len(blobs) / 20.0)

    if not blobs:
        # empty scene: the interior stats fall back to the whole photo,
        # so "plain table" is learnable rather than an all-zero row
        region = np.ones(gray.shape, bool)
        f["circ_med"] = f["solid_med"] = f["area_med"] = 0.0
    else:
        circ = [grader.circularity(c) for c in blobs]
        sol = [grader.solidity(c) for c in blobs]
        areas = [cv2.contourArea(c) for c in blobs]
        f["circ_med"] = float(np.median(circ))
        f["solid_med"] = float(np.median(sol))
        f["area_med"] = float(np.median(areas) / img_area)
        m = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m, blobs, -1, 255, -1)
        k = max(3, int(0.01 * min(h, w))) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inner = cv2.erode(m, kern) > 0
        region = inner if inner.sum() > 500 else (m > 0)

    H = hsv[:, :, 0].astype(np.float32)[region]
    S = hsv[:, :, 1].astype(np.float32)[region]
    V = hsv[:, :, 2].astype(np.float32)[region]
    G = gray.astype(np.float32)[region]
    n = max(1, H.size)

    f["h_med"] = float(np.median(H)) / 180.0
    f["h_std"] = float(H.std()) / 90.0
    f["s_med"] = float(np.median(S)) / 255.0
    f["s_std"] = float(S.std()) / 128.0
    f["v_med"] = float(np.median(V)) / 255.0
    f["v_std"] = float(V.std()) / 128.0
    f["g_std"] = float(G.std()) / 128.0

    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
    f["tex_var"] = float(min(1.0, lap[region].var() / 2000.0))

    f["green"] = float(((H >= 40) & (H <= 90) & (S >= 60) & (V >= 60)).sum() / n)
    f["cyan_blue"] = float(((H >= 90) & (H <= 135) & (S >= 70)).sum() / n)
    f["magenta"] = float(((H >= 135) & (H < 168) & (S >= 70)).sum() / n)
    f["gloss"] = float(((V > 235) & (S < 60)).sum() / n)
    f["neutral_white"] = float(((S < 25) & (V > 210)).sum() / n)
    f["vivid"] = float(((V >= 205) & (S >= 165)).sum() / n)
    # tomato / chilli red: hue at both ends of the wheel, very saturated
    f["red_frac"] = float((((H <= 8) | (H >= 172)) & (S >= 140)
                           & (V >= 90)).sum() / n)

    f["lab_a"] = float(np.median(lab[:, :, 1].astype(np.float32)[region])) / 255.0
    f["lab_b"] = float(np.median(lab[:, :, 2].astype(np.float32)[region])) / 255.0

    edges = cv2.Canny(gray, 60, 160)
    f["edge_den"] = float((edges[region] > 0).mean())
    return f, len(blobs)


def vec(f):
    return [float(f[k]) for k in FEATURE_ORDER]


# ----------------------------------------------------------------------------
# fallback (only if the model file is missing)
# ----------------------------------------------------------------------------
def _heuristic(f, n_blobs):
    """Conservative colour/shape rules - used ONLY when the trained
    model file is missing. Says 'not onion' only for clear cases."""
    if n_blobs == 0:
        return False, 0.85, "no object of onion size found in the photo"
    if f["green"] > 0.45:
        return False, 0.7, "green surface (leaf / vegetable, not onion skin)"
    if f["cyan_blue"] > 0.20:
        return False, 0.7, "blue surface (not an onion colour)"
    if f["red_frac"] > 0.55 and f["g_std"] < 0.15:
        return False, 0.65, "smooth bright red surface (tomato-like)"
    return True, 0.6, "looks like onion skin (rule fallback, model missing)"


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def check(bgr, threshold=0.5):
    """Is there an onion in this image?

    Returns a dict:
        is_onion   True / False
        confidence probability of the winning answer (0..1)
        p_onion    probability of the "onion" class
        reason     plain-English explanation for the user
        model      which engine answered
    """
    try:
        f, n_blobs = features(bgr)
    except Exception as exc:
        return {"is_onion": True, "confidence": 0.0, "p_onion": 0.0,
                "reason": f"presence check skipped ({exc})",
                "model": "unavailable", "n_blobs": 0}

    m = load_model()
    if not m:
        ok, conf, why = _heuristic(f, n_blobs)
        return {"is_onion": ok, "confidence": round(conf, 3),
                "p_onion": round(conf if ok else 1 - conf, 3),
                "reason": why, "model": "heuristic-fallback",
                "n_blobs": n_blobs}

    p = _proba(m, vec(f))
    classes = list(m["classes"])
    p_onion = float(p[classes.index("onion")]) if "onion" in classes else 0.0
    is_onion = p_onion >= threshold
    if is_onion:
        why = f"onion detected (scikit-learn confidence {p_onion:.0%})"
    else:
        why = _why_not(f, n_blobs, p_onion)
    return {"is_onion": is_onion,
            "confidence": round(p_onion if is_onion else 1 - p_onion, 3),
            "p_onion": round(p_onion, 3), "reason": why,
            "model": "sklearn-rf-json", "n_blobs": n_blobs,
            "trained_on": m.get("meta", {}).get("samples")}


def _why_not(f, n_blobs, p_onion):
    """A short, TRUE reason to show the user next to 'not found'."""
    bits = []
    if n_blobs == 0:
        bits.append("no onion-sized object stands out from the background")
    if f["green"] > 0.35:
        bits.append("the surface is green")
    if f["red_frac"] > 0.40:
        bits.append("the surface is bright tomato-red")
    if f["cyan_blue"] > 0.15:
        bits.append("the surface is blue")
    if f["magenta"] > 0.30:
        bits.append("the surface is purple")
    if f["gloss"] > 0.10 and f["g_std"] < 0.12:
        bits.append("the surface is smooth and glossy, not papery")
    detail = "; ".join(bits) if bits else \
        "the surface texture and colour do not match onion skin"
    return f"{detail} (onion probability {p_onion:.0%})"


def info():
    """Honest description of the active presence model (for reports)."""
    m = load_model()
    if not m:
        return {"name": "heuristic-fallback",
                "note": "models/onion_presence.json missing - "
                        "train it with train_presence.py"}
    meta = m.get("meta", {})
    return {"name": meta.get("model", "sklearn-random-forest"),
            "trees": len(m.get("trees", [])),
            "samples": meta.get("samples"),
            "holdout_accuracy": meta.get("holdout_accuracy"),
            "note": meta.get("note", "")}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python onion_presence.py photo.jpg [more.jpg ...]")
        raise SystemExit(0)
    print("model:", json.dumps(info()))
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        v = check(img)
        tag = "ONION" if v["is_onion"] else "NOT FOUND"
        print(f"{tag:9} p={v['p_onion']:.2f}  {os.path.basename(path)}"
              f"  - {v['reason']}")
