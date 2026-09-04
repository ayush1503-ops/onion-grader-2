#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 sweep_clf.py - honest hyperparameter search for the onion classifier.

 Tries small variations of the random forest and measures, for EACH
 variation:
   * LOPO accuracy on the 9 real non-sprout photos (the honest number)
   * false-ROTTEN count on the healthy red pile (must be low)
   * selftest-style agreement (mti holdout)
   * end-to-end score on the 12 real photos (in-sample for 2 - shown
     for completeness, NOT the selection criterion)

 Selection rule (fixed before looking): highest LOPO; ties broken by
 fewer pile false-ROTTEN, then selftest safety.
"""

import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import grader
import train_classifier as tc

CACHE = "/tmp/clf_dataset.pkl"


def forest_to_dict(rf, meta):
    trees = []
    for est in rf.estimators_:
        t = est.tree_

        def node(i):
            if t.children_left[i] == -1:
                return [-1, t.value[i][0].tolist()]
            return [int(t.feature[i]), float(t.threshold[i]),
                    node(t.children_left[i]), node(t.children_right[i])]
        trees.append(node(0))
    return {"format": 1, "features": tc.FEATURES,
            "classes": list(rf.classes_), "trees": trees, "meta": meta}


def install_model(d):
    """put a model dict directly into grader (no file round-trip)."""
    grader._CLF = d
    grader._CLF_TRIED = True


def uninstall_model():
    grader._CLF = None
    grader._CLF_TRIED = False


def lopo(X, y, groups, kinds, msl, md, rw):
    real_groups = [g for g in dict.fromkeys(groups[kinds == "real"])]
    ok = 0
    details = []
    for held in real_groups:
        te = groups == held
        tr = ~te
        w = np.where(kinds[tr] == "real", rw, 1.0)
        rf = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=msl, max_depth=md,
            class_weight="balanced_subsample", random_state=42,
            n_jobs=-1).fit(X[tr], y[tr], sample_weight=w)
        pred = rf.predict(X[te])
        pv = Counter(pred).most_common(1)[0][0]
        tv = Counter(y[te]).most_common(1)[0][0]
        ok += pv == tv
        details.append((held, pv, tv))
    return ok, details


def main():
    if os.path.exists(CACHE):
        samples, holdout = pickle.load(open(CACHE, "rb"))
    else:
        samples, holdout = tc.build_dataset()
        pickle.dump((samples, holdout), open(CACHE, "wb"))
    X = np.array([tc.vec(f) for f, *_ in samples])
    y = np.array([lab for _, lab, *_ in samples])
    groups = np.array([g for _, _, g, _ in samples])
    kinds = np.array([k for *_, k in samples])
    hx = np.array([tc.vec(f) for f, _ in holdout])
    hy = np.array([lab for _, lab in holdout])

    # evaluation photos (loaded once)
    pile1 = grader._fit_width(cv2.imread(
        "image-search/pile-of-red-onions-on-jute-sack-at-india-1.jpg"))
    real12 = tc.REAL

    print(f"{'cfg':<28}{'LOPO':>5}{'pile1R':>7}{'hold':>6}{'real12':>7}")
    results = []
    for msl in (2, 3, 4):
        for md in (None, 12):
            for rw in (1.0, 3.0):
                cfg = f"msl={msl} md={md} rw={rw}"
                lok, _det = lopo(X, y, groups, kinds, msl, md, rw)
                # final model on everything
                w = np.where(kinds == "real", rw, 1.0)
                rf = RandomForestClassifier(
                    n_estimators=250, min_samples_leaf=msl, max_depth=md,
                    class_weight="balanced_subsample", random_state=42,
                    n_jobs=-1).fit(X, y, sample_weight=w)
                # install into grader and measure behaviour
                install_model(forest_to_dict(rf, {}))
                rep = grader.analyze(pile1, out_dir=None)
                p1r = rep["class_counts"].get("ROTTEN", 0) + \
                    rep["class_counts"].get("DAMAGED", 0)
                hok = int((rf.predict(hx) == hy).sum())
                # end-to-end on the 12 real photos
                r12 = 0
                for truth, fname in real12:
                    img = cv2.imread(os.path.join("image-search", fname))
                    rep = grader.analyze(img, out_dir=None)
                    got = (rep["onions"][0]["label"]
                           if rep["onions"] else "?")
                    exp = "GOOD" if truth == "FRESH" else truth
                    r12 += got == exp
                uninstall_model()
                print(f"{cfg:<28}{lok:>3}/9{p1r:>7}{hok:>4}/16{r12:>5}/12")
                results.append((lok, -p1r, hok, msl, md, rw))
    results.sort(reverse=True)
    print("\nbest config:", results[0])


if __name__ == "__main__":
    main()
