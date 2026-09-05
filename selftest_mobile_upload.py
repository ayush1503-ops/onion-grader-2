#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 selftest_mobile_upload.py - the upload path must survive PHONES
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

Phone browsers do NOT post files the tidy way a desktop does. This test
replays the awkward-but-real payloads that used to break the app, so the
"upload does nothing on my phone" bug cannot come back silently.

WHAT IT REPLAYS
  1  a normal .jpg upload                                 -> must GRADE
  2  a blob with NO filename          (iOS Safari camera) -> must GRADE
  3  filename "blob", octet-stream    (Android WebView)   -> must GRADE
  4  field name "file" / "image"      (share targets)     -> must GRADE
  5  an .HEIC filename with JPEG bytes(browser-converted) -> must GRADE
  6  an UPPERCASE .JPG extension                          -> must GRADE
  7  a PNG screenshot                                     -> must GRADE
  8  real HEIC bytes                  (iPhone default)    -> must GRADE
  9  an EXIF-rotated portrait photo                       -> must GRADE
 10  zero bytes (interrupted upload)                      -> must FAIL clearly
 11  a text file                                          -> must FAIL clearly
 12  no file at all                                       -> must FAIL clearly

RUN (the app must already be running on :8000)
    python app.py &
    python selftest_mobile_upload.py
    python selftest_mobile_upload.py --url http://192.168.1.5:8000
"""

import argparse
import io
import sys

import cv2
import numpy as np

try:
    import requests
except ImportError:
    sys.exit("This test needs 'requests':  pip install requests")

from PIL import Image

# A synthetic scene with a KNOWN, stable count (8 onions). Real heap
# photos are counted +/- a couple of onions run to run (they are piles -
# grader.py says so itself), which would make this test flaky for
# reasons that have nothing to do with the upload path.
PHOTO = "test_images/test_batch_1.jpg"
EXPECT_ONIONS = 8


def jpeg_with_orientation(src, tag):
    """Store the photo the way a phone does: pixels rotated, EXIF tag set."""
    base = Image.open(src).convert("RGB")
    # tag 6 = "rotate 90 CW to display", so stored pixels are 90 CCW
    op = {3: Image.ROTATE_180, 6: Image.ROTATE_90, 8: Image.ROTATE_270}[tag]
    stored = base.transpose(op)
    exif = stored.getexif()
    exif[274] = tag
    buf = io.BytesIO()
    stored.save(buf, "JPEG", exif=exif.tobytes(), quality=92)
    return buf.getvalue()


def heic_bytes(src):
    """Real HEIC bytes, or None when pillow-heif is not installed."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        buf = io.BytesIO()
        Image.open(src).convert("RGB").save(buf, "HEIF", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def cases(raw, png, rot, heic):
    """(label, files-dict, must_succeed)"""
    out = [
        ("normal .jpg", {"photo": ("photo.jpg", raw, "image/jpeg")}, True),
        ("blob, NO filename (iOS Safari)",
         {"photo": ("", raw, "image/jpeg")}, True),
        ("filename 'blob', octet-stream (Android)",
         {"photo": ("blob", raw, "application/octet-stream")}, True),
        ("field name 'file'", {"file": ("blob", raw, "image/jpeg")}, True),
        ("field name 'image', no filename",
         {"image": ("", raw, "image/jpeg")}, True),
        (".HEIC filename, JPEG bytes",
         {"photo": ("IMG_0421.HEIC", raw, "image/heic")}, True),
        ("UPPERCASE .JPG",
         {"photo": ("IMG_0421.JPG", raw, "image/jpeg")}, True),
        ("PNG screenshot",
         {"photo": ("Screenshot.png", png, "image/png")}, True),
        ("EXIF-rotated portrait (orientation 6)",
         {"photo": ("IMG_rot.jpg", rot, "image/jpeg")}, True),
        ("zero bytes (interrupted upload)",
         {"photo": ("photo.jpg", b"", "image/jpeg")}, False),
        ("a text file, not a photo",
         {"photo": ("notes.txt", b"hello world", "text/plain")}, False),
    ]
    if heic:
        out.insert(8, ("real HEIC bytes (iPhone default)",
                       {"photo": ("IMG_9001.heic", heic, "image/heic")}, True))
    return out


def main():
    ap = argparse.ArgumentParser(description="mobile upload regression test")
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    endpoint = args.url.rstrip("/") + "/api/analyze"

    raw = open(PHOTO, "rb").read()
    png = cv2.imencode(".png", cv2.imread(PHOTO))[1].tobytes()
    rot = jpeg_with_orientation(PHOTO, 6)
    heic = heic_bytes(PHOTO)
    if heic is None:
        print("note: pillow-heif not installed - skipping the real-HEIC case")

    fails = 0
    for label, files, must_ok in cases(raw, png, rot, heic):
        try:
            r = requests.post(endpoint, files=files,
                              data={"mode": "cv"}, timeout=180)
            body = r.json()
        except Exception as exc:
            print(f"  FAIL  {label:42} request error: {exc}")
            fails += 1
            continue
        got_ok = bool(body.get("ok"))
        good = (got_ok == must_ok)
        # a successful upload must also find the onions we know are there:
        # that proves the BYTES survived the trip (right format, upright)
        if got_ok and body.get("rep", {}).get("onion_count") != EXPECT_ONIONS:
            good = False
        # a failure must still explain itself in plain words
        if not got_ok and not body.get("error"):
            good = False
        fails += 0 if good else 1
        detail = (f"onions={body.get('rep', {}).get('onion_count')}" if got_ok
                  else str(body.get("error", ""))[:52])
        print(f"  {'PASS' if good else 'FAIL'}  {label:42} "
              f"want={'grade' if must_ok else 'clear error':11} {detail}")

    # no file at all
    try:
        body = requests.post(endpoint, data={"mode": "cv"}, timeout=60).json()
        good = (not body.get("ok")) and bool(body.get("error"))
        fails += 0 if good else 1
        print(f"  {'PASS' if good else 'FAIL'}  {'no file at all':42} "
              f"want={'clear error':11} {str(body.get('error'))[:52]}")
    except Exception as exc:
        print(f"  FAIL  {'no file at all':42} {exc}")
        fails += 1

    print(f"\n{'ALL MOBILE UPLOAD CHECKS PASSED' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
