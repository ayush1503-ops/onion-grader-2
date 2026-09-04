#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 selftest.py - automatic detection check on the synthetic test set
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

Runs grader.py on every photo in test_images/ and compares the result
with the ground truth printed by make_test_images.py (the images are
computer-generated at KNOWN sizes, so the correct answer is known):

  * onion COUNT per photo
  * class counts (GOOD / DAMAGED / ROTTEN / SPROUTED / UNDERSIZED)
  * the mm SCALE source (coin / EXIF camera-distance / honest guess)
  * measured diameters where the photo allows a fair measurement

HONESTY: passing this selftest only proves the CODE works on these
SYNTHETIC images. Real-world accuracy must still be measured on real
photos labeled by a human expert.

RUN:  python selftest.py        (exit code 0 = all checks passed)
"""

import os
import sys
import tempfile

import grader

# ground truth, exactly as printed by make_test_images.py
CASES = [
    # (file, analyze kwargs, count, class counts, scale source must
    #  contain, expected sorted diameters in mm or None)
    ("test_batch_1.jpg", {"coin_mm": 27.0}, 8,
     {"GOOD": 4, "DAMAGED": 1, "ROTTEN": 1, "SPROUTED": 1,
      "UNDERSIZED": 1}, "coin", None),
    ("test_batch_2_touching.jpg", {"coin_mm": 27.0}, 6,
     {"GOOD": 3, "DAMAGED": 1, "ROTTEN": 1, "SPROUTED": 1}, "coin", None),
    ("test_batch_3_no_coin.jpg", {}, 4,
     {"GOOD": 4}, "assumed", None),
    ("test_batch_4_dark.jpg", {"coin_mm": 27.0}, 6,
     {"GOOD": 3, "DAMAGED": 1, "ROTTEN": 1, "SPROUTED": 1}, "coin", None),
    ("test_batch_5_pile.jpg", {"coin_mm": 27.0}, 8,
     # the covered back onion shows only CLEAN skin -> visible-surface
     # honesty makes it GOOD; the pile test is about count + splitting
     {"GOOD": 8}, "coin", None),
    ("test_batch_6_exif.jpg", {"distance_mm": 400.0}, 4,
     {"GOOD": 4}, "EXIF", [40.0, 55.0, 55.0, 55.0]),
    ("test_batch_7_uneven.jpg", {}, 4,
     {"GOOD": 4}, None, [55.0, 55.0, 55.0, 55.0]),
]

MM_TOL = 0.10          # measured diameter within 10% ...
MM_TOL_MIN = 3.0       # ... but at least +/- 3 mm (pixel rounding)


def check(case, out_dir):
    fname, kwargs, count, classes, scale_sub, diameters = case
    path = os.path.join("test_images", fname)
    rep = grader.analyze(path, out_dir=out_dir, **kwargs)
    fails = []

    if rep["onion_count"] != count:
        fails.append(f"count {rep['onion_count']} != {count}")

    got_cls = rep["class_counts"]
    for cls, want in classes.items():
        if got_cls.get(cls, 0) != want:
            fails.append(f"{cls} {got_cls.get(cls, 0)} != {want}")

    if scale_sub and scale_sub not in rep["scale_source"]:
        fails.append(f"scale '{rep['scale_source']}' lacks '{scale_sub}'")

    if diameters:
        got = sorted(o["diameter_mm"] for o in rep["onions"])
        for g, e in zip(got, sorted(diameters)):
            if abs(g - e) > max(MM_TOL * e, MM_TOL_MIN):
                fails.append(f"diameter {g:.1f} mm != {e:.0f} mm")
    return rep, fails


def main():
    if not os.path.isdir("test_images"):
        sys.exit("run this from the project root (next to test_images/)")
    print("OnionGrader detection selftest (SYNTHETIC images)\n"
          + "-" * 62)
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="onion_selftest_") as tmp:
        for case in CASES:
            rep, fails = check(case, tmp)
            ok = not fails
            all_ok &= ok
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {case[0]:<28} onions={rep['onion_count']:<2} "
                  f"scale={rep['scale_source']}")
            for f in fails:
                print(f"        - {f}")
    print("-" * 62)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("(synthetic demo data - real accuracy needs a real test set)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
