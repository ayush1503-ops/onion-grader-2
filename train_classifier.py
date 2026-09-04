#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 train_classifier.py - train the ONION SURFACE CLASSIFIER (advanced ML)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT IT DOES (simple words):
    Trains a Random Forest to look at one onion's measured features
    (colours, local dark patches, texture, saturation...) and decide:
    GOOD, DAMAGED or ROTTEN. The trained model is exported to
    models/onion_clf.json as plain numbers - grader.py reads that file
    with ONLY numpy (no scikit-learn needed at run time!), so the
    trained model also works on the Vercel deployment.

HOW IT LEARNS (3 data sources - all real, nothing invented):
    1. 12 real labelled web photos  (image-search/) - each augmented
       with flips / rotations / brightness changes so one photo gives
       many slightly different training samples.
    2. The 3 red-onion pile photos, used as GOOD samples. HONEST
       ASSUMPTION: stock/market photos show healthy onions. This is
       what teaches the model that DARK RED skin is NOT rot.
    3. Synthetic onions drawn by code (yellow / red / white varieties,
       with and without bruises, rot patches, sprouts). Labels are
       correct BY CONSTRUCTION. These teach the basic physics of the
       features without needing thousands of real photos.

HONEST EVALUATION (never skip):
    Leave-One-Photo-Out (LOPO) on the 12 real photos: for each photo,
    the model is trained WITHOUT it and then tested on it. This is the
    honest number for "how good is this on new real photos" - with only
    12 photos it is a SMALL sample, and the script says so.

    SPROUTED is detected by the measured green_top rule BEFORE the
    model runs (a sprout is a vivid green shoot at the top - a clear,
    explainable signal), so sprout photos are evaluated on that rule.

RUN:  python train_classifier.py
=====================================================================
"""

import json
import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import grader

RNG = np.random.default_rng(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "models", "onion_clf.json")

# the real labelled photos (same list as evaluate_real.py)
REAL = [
    ("GOOD",    "fresh-healthy-onion-whole-bulb-on-plain--1.jpg"),
    ("GOOD",    "fresh-healthy-onion-whole-bulb-on-plain--2.jpg"),
    ("GOOD",    "fresh-healthy-red-onion-whole-bulb-2.jpg"),
    ("DAMAGED", "damaged-bruised-onion-with-brown-blemish-1.jpg"),
    ("DAMAGED", "damaged-bruised-onion-with-brown-blemish-2.jpg"),
    ("ROTTEN",  "rotten-spoiled-onion-with-mold-dark-spot-1.jpg"),
    ("ROTTEN",  "rotten-spoiled-onion-with-mold-dark-spot-2.jpg"),
    ("ROTTEN",  "black-moldy-rotten-onion-closeup-1.jpg"),
    ("ROTTEN",  "black-moldy-rotten-onion-closeup-2.jpg"),
    # sprouted photos: the green_top RULE handles them (checked separately)
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-1.jpg"),
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-2.jpg"),
    ("SPROUTED", "sprouting-onion-with-green-shoots-growin-3.jpg"),
]
PILES = [  # assumed HEALTHY (stock/market photos) -> GOOD
    "pile-of-red-onions-on-jute-sack-at-india-1.jpg",
    "pile-of-red-onions-on-jute-sack-at-india-2.jpg",
    "pile-of-red-onions-on-jute-sack-at-india-3.jpg",
]
FEATURES = grader.FEATURE_ORDER_V3


# ------------------------------------------------------------------
# feature extraction helpers (use the app's own pipeline = no cheating)
# ------------------------------------------------------------------
def onion_contours(bgr):
    """run the app's own segmentation, return (gray, hsv, [contours])."""
    gray, mask = grader.make_object_mask(bgr)
    blobs = grader.get_blobs(mask)
    return gray, mask, blobs


def features_of_blob(gray, hsv, contour):
    omask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(omask, [contour], -1, 255, -1)
    return grader.onion_features(gray, hsv, omask)


def vis_of(contour, all_contours=None):
    """visibility of one onion (extent = area / convex-hull area), same
    formula the app uses. For pile onions the app computes it over the
    WHOLE detected set, so we do too when all_contours is given."""
    if all_contours is not None:
        vals = grader.compute_visibility(all_contours)
        try:
            return float(vals[list(all_contours).index(contour)])
        except (ValueError, IndexError):
            return 1.0
    return float(grader.compute_visibility([contour])[0])


def augment(bgr, i):
    """deterministic augmentations: flips + 90-degree rotations +
    brightness scaling. Returns a list of (variant_id, image)."""
    outs = []
    for flip in (None, 1):
        for rot in (None, cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            img = bgr
            if flip is not None:
                img = cv2.flip(img, flip)
            if rot is not None:
                img = cv2.rotate(img, rot)
            outs.append((i, img, 1.0))          # original brightness
            outs.append((i, img, 0.78))         # darker variant
            outs.append((i, img, 1.22))         # brighter variant
    return outs


def scale_brightness(bgr, f):
    img = np.clip(bgr.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------
# synthetic onion generator (labels correct by construction)
# ------------------------------------------------------------------
def hsv_bgr(h, s, v):
    col = cv2.cvtColor(np.array([[[int(h) % 180, int(s), int(v)]]],
                                  np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(col[0]), int(col[1]), int(col[2]))   # plain tuple


def synth_onion(kind):
    """draw ONE onion on a light canvas. kind: good/damaged/rotten/sprouted.
    Varieties: yellow (like the app's old test images), dark RED (the
    variety that used to be falsely graded rotten) and whitish."""
    W, H = 560, 440
    bg = int(RNG.integers(200, 235))          # light tray / paper
    img = np.full((H, W, 3), (bg, bg, bg), np.uint8)

    variety = RNG.choice(["yellow", "red", "white"], p=[.4, .4, .2])
    if variety == "yellow":
        hue, s0, v0 = 13 + int(RNG.integers(-3, 4)), 108, 185
    elif variety == "red":
        # dark red-purple skin. NOTE: measured on real photos, red onion
        # skin reads hue 5-20 in OpenCV HSV (purple-brown) with HIGH
        # saturation - that is exactly why absolute "brown" thresholds
        # used to fail. We draw it the way the camera sees it.
        hue = int(RNG.integers(5, 21))
        s0, v0 = int(RNG.integers(120, 165)), int(RNG.integers(105, 150))
    else:                                      # whitish onion
        hue, s0, v0 = 16, int(RNG.integers(40, 75)), 190

    cx, cy = int(RNG.integers(150, W - 150)), int(RNG.integers(150, H - 120))
    r = int(RNG.integers(60, 105))

    # body with smooth radial shading (centre brighter than the rim)
    for i in range(24):
        t = i / 23.0
        rr = int(r * (1.0 - 0.75 * t))
        v = int(np.clip(v0 + 28 * t, 0, 255))
        cv2.circle(img, (cx, cy), max(1, rr),
                   hsv_bgr(hue, s0, v), -1)
    # darker rim (a real onion edge)
    cv2.circle(img, (cx, cy), r - 1, hsv_bgr(hue, min(179, s0 + 15),
               int(np.clip(v0 - 18, 0, 255))), 3, cv2.LINE_AA)
    # papery skin arcs (texture, still healthy-bright)
    for _ in range(5):
        a = int(RNG.integers(0, 180))
        cv2.ellipse(img, (cx, cy), (int(r * .85), int(r * .6)), a, 0, 100,
                    hsv_bgr(hue, s0 + 8, int(np.clip(v0 - 12, 0, 255))),
                    2, cv2.LINE_AA)

    def blotch(rr, h, s, v, ox=0, oy=0):
        """irregular defect blob (same idea as make_test_images.blotch)"""
        pts = []
        for k in range(14):
            ang = k / 14 * 2 * np.pi
            rad = rr * (0.75 + 0.5 * RNG.random())
            pts.append((cx + ox + rad * np.cos(ang),
                        cy + oy + rad * np.sin(ang)))
        pts = np.array(pts, np.int32)
        col = hsv_bgr(h, s, v)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    if kind == "damaged":
        # bruise: a LOCAL brown patch, clearly darker than its
        # neighbourhood but not black. Three severities: clear bruise,
        # a LIGHT bruise, and a light bruise on an overall DULL skin
        # (real bruised onions often lose their glossy saturation).
        style = RNG.random()
        if style < 0.4:
            blotch(0.45 * r, (hue + 3) % 180, min(179, s0 + 25),
                   max(35, v0 - int(RNG.integers(40, 60))),
                   ox=int(0.2 * r), oy=int(-0.1 * r))
            cv2.line(img, (cx - int(.5 * r), cy - int(.3 * r)),
                     (cx + int(.4 * r), cy + int(.5 * r)),
                     hsv_bgr(10, 110, max(30, v0 - 130)), 6, cv2.LINE_AA)
        elif style < 0.7:
            blotch(0.35 * r, (hue + 4) % 180, min(179, s0 + 18),
                   max(45, v0 - int(RNG.integers(28, 42))),
                   ox=int(-0.1 * r), oy=int(0.2 * r))
        else:
            # dull skin + light bruise: saturation slightly dropped and
            # the skin is a bit darker (soft damaged skin loses its
            # gloss but not all of it - measured on a real photo)
            hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_img[:, :, 1] *= float(RNG.uniform(0.72, 0.95))
            hsv_img[:, :, 2] *= float(RNG.uniform(0.80, 0.95))
            img[:] = cv2.cvtColor(
                np.clip(hsv_img, 0, 255).astype(np.uint8),
                cv2.COLOR_HSV2BGR)
            blotch(0.30 * r, (hue + 4) % 180, min(179, int(s0 * 0.8)),
                   max(50, v0 - int(RNG.integers(25, 40))),
                   ox=int(0.15 * r), oy=int(0.1 * r))
    if kind == "rotten":
        # rot: four visual styles, all real:
        #  a) black/gray mold - very dark, low saturation
        #  b) dull brown soggy mold - mid-dark patch, matte
        #  c) whole-skin dull brown mold - no local patch at all, the
        #     entire surface is matte dull brown (real photo case)
        #  d) gray-black DESATURATED patches - measured on a real
        #     photo where the key signal is desat_dark, not darkness
        style = RNG.random()
        if style < 0.4:
            blotch(0.45 * r, 12, int(RNG.integers(35, 95)),
                   int(RNG.integers(25, 55)),
                   ox=int(-0.15 * r), oy=int(0.1 * r))
            blotch(0.18 * r, 12, 45, int(RNG.integers(25, 50)),
                   ox=int(0.4 * r), oy=int(-0.35 * r))
        elif style < 0.6:
            blotch(0.62 * r, int(RNG.integers(9, 15)),
                   int(RNG.integers(55, 105)),
                   int(RNG.integers(65, 110)),
                   ox=int(RNG.integers(-10, 10)), oy=int(RNG.integers(-10, 10)))
        elif style < 0.8:
            # whole-skin dull matte brown (light or darker variant)
            v0 = int(RNG.integers(95, 160)); s0 = int(RNG.integers(60, 110))
            hue = int(RNG.integers(8, 19))
            for i in range(24):
                t = i / 23.0
                rr = int(r * (1.0 - 0.75 * t))
                v = int(np.clip(v0 + 22 * t, 0, 255))
                cv2.circle(img, (cx, cy), max(1, rr),
                           hsv_bgr(hue, s0, v), -1)
        else:
            # gray-black desaturated patches (the desat_dark signal)
            blotch(0.42 * r, int(RNG.integers(8, 16)),
                   int(RNG.integers(20, 55)),
                   int(RNG.integers(40, 85)),
                   ox=int(-0.1 * r), oy=int(0.15 * r))
            blotch(0.25 * r, 12, int(RNG.integers(20, 50)),
                   int(RNG.integers(40, 80)),
                   ox=int(0.3 * r), oy=int(-0.25 * r))
    if kind == "sprouted":
        cv2.ellipse(img, (cx, cy - int(0.85 * r)),
                    (max(3, int(0.16 * r)), max(3, int(0.30 * r))),
                    0, 0, 360, hsv_bgr(60, 160, 150), -1)

    # light sensor noise + JPEG artefacts (so features are realistic)
    img = np.clip(img.astype(np.int16) + RNG.integers(-6, 7, img.shape),
                  0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def synth_sample(kind):
    """draw a synthetic onion, segment it with the app's own pipeline,
    and return its feature vector + green_top (or None if not found)."""
    img = synth_onion(kind)
    gray, mask, blobs = onion_contours(img)
    if not blobs:
        return None
    c = max(blobs, key=cv2.contourArea)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = features_of_blob(gray, hsv, c)
    f["vis"] = vis_of(c)
    return f


# ------------------------------------------------------------------
# dataset building
# ------------------------------------------------------------------
def analyze_to_list(img, max_onions=1):
    """Run the app's OWN full pipeline (segmentation, watershed splits,
    fragment filter, feature engine, visibility) on one image and
    return the feature dicts of its biggest onion(s). This guarantees
    training features are EXACTLY what classify() sees at run time -
    no train/serve mismatch."""
    try:
        rep = grader.analyze(img, out_dir=None)     # in-memory mode
    except Exception:
        return []
    onions = sorted(rep.get("onions", []),
                    key=lambda o: o.get("area_px", 0), reverse=True)
    return [o["features"] for o in onions[:max_onions]]


def build_dataset():
    samples = []          # (features dict, label, group, source)
    print("building dataset ...")

    # ---- 1) real labelled singles (augmented) ----
    for label, fname in REAL:
        if label == "SPROUTED":        # handled by the green_top rule
            continue
        bgr = cv2.imread(os.path.join("image-search", fname))
        if bgr is None:
            print(f"  !! missing {fname}")
            continue
        for gi, img, br in augment(bgr, len(samples)):
            im2 = img if br == 1.0 else scale_brightness(img, br)
            for f in analyze_to_list(im2, 1):
                samples.append((f, label, fname, "real"))
    n_real = len(samples)
    print(f"  real singles (augmented): {n_real}")

    # ---- 2) healthy red piles -> GOOD (assumption, disclosed) ----
    n_pile = 0
    for fname in PILES:
        bgr = grader._fit_width(cv2.imread(os.path.join("image-search", fname)))
        for variant in (1.0, 0.85, 1.12):
            im2 = bgr if variant == 1.0 else scale_brightness(bgr, variant)
            for f in analyze_to_list(im2, 10):
                samples.append((f, "GOOD", fname, "pile"))
                n_pile += 1
    print(f"  healthy pile onions (assumed GOOD): {n_pile}")

    # ---- 3) synthetic onions (correct by construction) ----
    n_synth = 0
    for kind, n in [("good", 60), ("damaged", 40), ("rotten", 40)]:
        for _ in range(n):
            for f in analyze_to_list(synth_onion(kind), 1):
                samples.append((f, "GOOD" if kind == "good" else kind.upper(),
                                "synth", "synth"))
                n_synth += 1
    print(f"  synthetic onions: {n_synth}")

    # ---- 3b) the app's OWN test-image style (make_test_images.py) ----
    # Same drawing physics as the synthetic selftest set, but different
    # random instances. This makes sure the trained model agrees with
    # the known-correct labels of that rendering style.
    # The last 16 are a HOLDOUT: never trained on, only predicted -
    # an honest synthetic agreement check.
    import make_test_images as mti
    n_mti = 0
    holdout = []          # (features, label) - NOT added to samples
    plan = ([("good", 40), ("damaged", 16), ("rotten", 16)] +
            [("holdout-good", 8), ("holdout-damaged", 4), ("holdout-rotten", 4)])
    for kind, n in plan:
        hold = kind.startswith("holdout-")
        base = kind.split("-", 1)[1] if hold else kind
        for _ in range(n):
            img = mti.new_canvas()
            r = int(RNG.integers(45, 76))
            cx = int(RNG.integers(r + 40, img.shape[1] - r - 40))
            cy = int(RNG.integers(r + 40, img.shape[0] - r - 40))
            mti.draw_onion(img, cx, cy, r, base)
            noise = RNG.normal(0, RNG.uniform(2.5, 6.0),
                               img.shape[:2] + (1,))
            img = np.clip(img.astype(np.float32) + noise,
                          0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY,
                                    int(RNG.integers(88, 96))])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            lab = "GOOD" if base == "good" else base.upper()
            for f in analyze_to_list(img, 1):
                if hold:
                    holdout.append((f, lab))
                else:
                    samples.append((f, lab, "mti-synth", "synth"))
                    n_mti += 1
    print(f"  app test-image style: {n_mti} (+{len(holdout)} holdout)")
    return samples, holdout


def vec(f):
    return [float(f[k]) for k in FEATURES]


# ------------------------------------------------------------------
# export the trained forest as plain JSON (numpy-only at run time)
# ------------------------------------------------------------------
def export_forest(rf, meta, path):
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        def node(i):
            if t.children_left[i] == -1:
                return [-1, t.value[i][0].tolist()]      # leaf: class counts
            return [int(t.feature[i]), float(t.threshold[i]),
                    node(t.children_left[i]), node(t.children_right[i])]
        trees.append(node(0))
    obj = {"format": 1, "features": FEATURES,
           "classes": list(rf.classes_), "trees": trees, "meta": meta}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    print(f"exported {len(trees)} trees -> {path} "
          f"({os.path.getsize(path) // 1024} KB)")


def walk_tree(node, x):
    while node[0] != -1:
        node = node[2] if x[node[0]] <= node[1] else node[3]
    return node[1]          # class counts of the leaf


def json_predict(model, x):
    votes = np.zeros(len(model["classes"]))
    for tree in model["trees"]:
        counts = np.array(walk_tree(tree, x), dtype=float)
        votes += counts / max(1.0, counts.sum())    # sklearn: mean proba
    return model["classes"][int(np.argmax(votes))]


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    samples, holdout = build_dataset()
    X = np.array([vec(f) for f, *_ in samples])
    y = np.array([lab for _, lab, *_ in samples])
    groups = np.array([g for _, _, g, _ in samples])
    kinds = np.array([k for *_, k in samples])

    def make_rf():
        return RandomForestClassifier(
            n_estimators=250, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)

    # real photos get extra weight: they are the true target domain
    # (piles are assumed-healthy, synthetics are a simulator)
    real_w = np.where(kinds == "real", 3.0, 1.0)

    # ---------- HONEST EVALUATION: leave-one-real-photo-out ----------
    print("\nLOPO cross-validation on the 9 real non-sprout photos")
    print("-" * 62)
    real_groups = [g for g in dict.fromkeys(groups[kinds == "real"])]
    lopo_ok, lopo_n = 0, 0
    for held in real_groups:
        te = (groups == held)
        tr = ~te
        rf = make_rf().fit(X[tr], y[tr], sample_weight=real_w[tr])
        pred = rf.predict(X[te])
        truth = y[te]
        # photo-level = majority vote over its augmented samples
        from collections import Counter
        pv, tv = Counter(pred).most_common(1)[0][0], \
            Counter(truth).most_common(1)[0][0]
        ok = pv == tv
        lopo_ok += ok
        lopo_n += 1
        print(f"  [{'OK ' if ok else 'MISS'}] {held[:44]:<46} "
              f"-> {pv:<8} (truth {tv})")
    print(f"  LOPO model accuracy: {lopo_ok}/{lopo_n}")

    # sprout rule on the 3 sprout photos (measured, not trained)
    sprout_ok = 0
    for _, fname in REAL:
        if not fname.startswith("sprouting"):
            continue
        bgr = cv2.imread(os.path.join("image-search", fname))
        gray, mask, blobs = onion_contours(bgr)
        if blobs:
            c = max(blobs, key=cv2.contourArea)
            f = features_of_blob(gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), c)
            if f["green_top"] >= grader.GREEN_SPROUTED:
                sprout_ok += 1
    print(f"  sprout RULE (green_top>=0.10): {sprout_ok}/3")

    # ---------- FINAL model: trained on everything ----------
    rf = make_rf().fit(X, y, sample_weight=real_w)
    train_acc = rf.score(X, y)
    print(f"\nfinal model trained on all {len(y)} samples "
          f"(train accuracy {train_acc:.3f} - NOT a test number)")

    # honest synthetic holdout (generated, labelled, NEVER trained on)
    hold_ok, hold_n, hold_txt = 0, 0, ""
    if holdout:
        hx = np.array([vec(f) for f, _ in holdout])
        hy = np.array([lab for _, lab in holdout])
        hpred = rf.predict(hx)
        hold_ok, hold_n = int((hpred == hy).sum()), len(hy)
        for (f, lab), p in zip(holdout, hpred):
            if lab != p:
                print(f"   holdout miss: truth {lab} -> predicted {p}")
        print(f"synthetic HOLDOUT agreement (never trained on): "
              f"{hold_ok}/{hold_n}")
        hold_txt = f"{hold_ok}/{hold_n}"

    # export + verify the JSON model reproduces sklearn EXACTLY
    meta = {
        "model": "random-forest-250-v3",
        "features": len(FEATURES),
        "n_samples": int(len(y)),
        "sources": {"real_augmented": int((kinds == 'real').sum()),
                    "pile_assumed_good": int((kinds == 'pile').sum()),
                    "synthetic": int((kinds == 'synth').sum())},
        "eval": {"lopo_real_photos": f"{lopo_ok}/{lopo_n}",
                 "sprout_rule": f"{sprout_ok}/3",
                 "synthetic_holdout": hold_txt,
                 "note": ("LOPO = leave-one-real-photo-out, the honest "
                          "small-sample test. Train accuracy is NOT a "
                          "test. Pile photos are ASSUMED healthy.")},
    }
    export_forest(rf, meta, OUT_PATH)

    with open(OUT_PATH) as fh:
        model = json.load(fh)
    mism = 0
    for i in range(len(X)):
        if json_predict(model, X[i]) != rf.predict(X[i:i + 1])[0]:
            mism += 1
    print(f"JSON export vs sklearn on all {len(X)} samples: "
          f"{mism} mismatches " + ("(EXACT MATCH)" if mism == 0 else "!! FIX"))

    # class balance summary
    from collections import Counter
    print("class balance:", dict(Counter(y)))
    print("\nLOPO honest result: "
          f"{lopo_ok}/{lopo_n} real photos + sprout rule {sprout_ok}/3")


if __name__ == "__main__":
    main()
