#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 app.py - ONION QUALITY GRADING WEB APP  (run it on your laptop,
          then open it from your PHONE's browser - no install needed)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT IT DOES (simple words):
    1. Shows a web page where you pick a photo of onions
       (on a phone, the button opens the CAMERA).
    2. You tell it which coin is in the photo (the "ruler").
    3. It runs grader.py (count -> split touching -> measure -> classify
       -> grade) and shows the result right on the page:
       % Grade A / % URS / % Reject, a table for every onion,
       the annotated photo and the report card.
    4. All report files stay in outputs/ and can be downloaded.

HOW TO RUN:
    python app.py
    Then on this laptop:      http://localhost:8000
    On your phone (SAME Wi-Fi): the app prints a link like
    http://192.168.x.x:8000 - type that in the phone browser.

HONESTY: grades VISIBLE SURFACE quality only. A normal photo CANNOT
detect internal rot, internal damage or internal moisture. This is
shown on the web page and in every report.
"""

import os
import re
import socket
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import grader
import yolo_mode

# ------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVERLESS = bool(
    os.environ.get("VERCEL")
    or os.environ.get("VERCEL_ENV")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
)
if SERVERLESS:
    os.environ.setdefault("VERCEL", "1")
    os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")
    os.environ.setdefault("OUTPUT_DIR", "/tmp/outputs")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (
    4 * 1024 * 1024 if SERVERLESS else 16 * 1024 * 1024
)

# Same coin menu as grader.py
COIN_MENU = {"10": 27.0, "2": 27.0, "5": 23.0, "1": 22.0}


def resolve_coin(coin_preset, coin_custom, distance_cm=None, assume_mm=None):
    """
    Work out the scale WITHOUT asking the user every time:
      - coin_preset "custom" + a number = exact coin size (advanced)
      - coin_preset "distance"          = estimate from camera distance
      - coin_preset "assume"            = assume a standard onion size
      - anything else (default)         = auto: assume Rs.10/Rs.2 = 27 mm;
        the detector still finds the coin automatically if one is present.
    Returns (coin_mm, distance_mm, assume_mm, error).
    """
    if coin_preset == "custom":
        try:
            coin_mm = float(coin_custom)
        except (TypeError, ValueError):
            return None, None, None, "Type the coin diameter in mm."
        if not (5 <= coin_mm <= 100):
            return None, None, None, "Coin diameter must be 5-100 mm."
        return coin_mm, None, None, None
    if coin_preset == "distance":
        try:
            d = float(distance_cm) * 10.0        # cm -> mm
        except (TypeError, ValueError):
            return None, None, None, "Type the camera distance in cm."
        if not (10 <= d <= 2000):
            return None, None, None, "Distance must be 10-2000 cm."
        return 27.0, d, None, None
    if coin_preset == "assume":
        try:
            a = float(assume_mm) if assume_mm else 55.0
        except (TypeError, ValueError):
            return None, None, None, "Assumed size must be a number (mm)."
        if not (10 <= a <= 200):
            return None, None, None, "Assumed size must be 10-200 mm."
        return 27.0, None, a, None
    # default: auto (coin auto-detect at the standard 27 mm)
    return 27.0, None, None, None


def clean_report(rep):
    """Copy the report without drawing-only keys (numpy/contours)."""
    clean = {k: v for k, v in rep.items() if k not in ("onions", "files")}
    clean["onions"] = [{k: v for k, v in o.items() if k != "contour"}
                       for o in rep["onions"]]
    return clean


# ------------------------------------------------------------------
# Helper: run the whole pipeline on one upload
# ------------------------------------------------------------------
def run_pipeline(file_storage, coin_preset, coin_custom, batch_id, mode="cv",
                 distance_cm=None, assume_mm=None):
    """Save the upload -> run the pipeline -> build download URLs."""
    filename = secure_filename(file_storage.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return None, "Please upload a photo file (.jpg .jpeg .png .webp .bmp)"

    # unique name so two uploads never overwrite each other
    stamp = grader.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_name = f"{stamp}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    file_storage.save(save_path)

    coin_mm, dist_mm, assu_mm, err = resolve_coin(
        coin_preset, coin_custom, distance_cm, assume_mm)
    if err:
        return None, err
    batch_id = (batch_id or "").strip() or None   # empty -> auto batch id

    try:
        if mode == "yolo":
            rep = yolo_mode.analyze(save_path, coin_mm=coin_mm,
                                    batch_id=batch_id, out_dir=OUTPUT_DIR,
                                    distance_mm=dist_mm, assume_mm=assu_mm)
        else:
            rep = grader.analyze(save_path, coin_mm=coin_mm,
                                 batch_id=batch_id, out_dir=OUTPUT_DIR,
                                 distance_mm=dist_mm, assume_mm=assu_mm,
                                 coin_assumed=(coin_preset != "custom"))
    except yolo_mode.ModelNotTrained as exc:
        return None, str(exc)
    except Exception as exc:                       # friendly error page
        return None, f"Analysis failed: {exc}"

    urls = file_urls(rep)
    if os.environ.get("VERCEL") == "1":          # embed the photo too
        import base64 as _b64
        ext = os.path.splitext(save_name)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".bmp": "image/bmp"}.get(ext, "image/jpeg")
        with open(os.path.join(UPLOAD_DIR, save_name), "rb") as fh:
            urls["original"] = f"data:{mime};base64," + \
                _b64.b64encode(fh.read()).decode()
    else:
        urls["original"] = f"/uploads/{save_name}"
    return {"rep": clean_report(rep), "urls": urls}, None


def file_urls(rep):
    """Download links for the report files of an analysis.

    Serverless mode (VERCEL=1): the disk is ephemeral, so files are
    embedded straight into the JSON response as data: URIs instead of
    links that would break on the next request."""
    names = [os.path.basename(f) for f in rep.get("files", [])]
    if len(names) < 4:
        return {}
    if os.environ.get("VERCEL") == "1":
        import base64 as _b64
        mimes = {0: "image/jpeg", 1: "application/json",
                 2: "text/plain", 3: "image/jpeg", 4: "image/jpeg"}
        urls = {}
        for i, key in enumerate(["annotated", "json", "txt", "card", "full"]):
            if i >= len(names):
                break
            path = os.path.join(OUTPUT_DIR, names[i])
            with open(path, "rb") as fh:
                urls[key] = (f"data:{mimes[i]};base64,"
                             + _b64.b64encode(fh.read()).decode())
        return urls
    urls = {
        "annotated": f"/outputs/{names[0]}",
        "json":      f"/outputs/{names[1]}",
        "txt":       f"/outputs/{names[2]}",
        "card":      f"/outputs/{names[3]}",
    }
    if len(names) >= 5:
        urls["full"] = f"/outputs/{names[4]}"   # one JPEG with everything
    return urls


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    """The one and only page (form + JavaScript that renders results)."""
    with open(os.path.join(BASE_DIR, "app_page.html"), "r",
              encoding="utf-8") as fh:
        return fh.read()


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """The page sends the photo here. Returns JSON results."""
    photo = request.files.get("photo")
    if photo is None or photo.filename == "":
        return jsonify({"ok": False,
                        "error": "Please choose a photo first."}), 400
    result, err = run_pipeline(
        photo,
        request.form.get("coin_preset", "auto"),
        request.form.get("coin_custom", ""),
        request.form.get("batch_id", ""),
        mode=request.form.get("mode", "cv"),
        distance_cm=request.form.get("distance_cm", ""),
        assume_mm=request.form.get("assume_mm", ""),
    )
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, **result})


@app.route("/api/live", methods=["POST"])
def api_live():
    """LIVE MODE: the browser sends the current camera frame ~every
    second. We run the pipeline in memory (fast, no files) and return
    detections. The browser draws boxes on top of the video."""
    frame = request.files.get("frame")
    if frame is None:
        return jsonify({"ok": False, "error": "No frame received."}), 400
    data = frame.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"ok": False, "error": "Frame was not a valid image."}), 400

    coin_mm, dist_mm, assu_mm, err = resolve_coin(
        request.form.get("coin_preset", "auto"),
        request.form.get("coin_custom", ""),
        request.form.get("distance_cm", ""),
        request.form.get("assume_mm", ""))
    if err:
        return jsonify({"ok": False, "error": err}), 400
    mode = request.form.get("mode", "cv")

    try:
        if mode == "yolo":
            rep = yolo_mode.analyze_frame(bgr, coin_mm=coin_mm)
        else:
            rep = grader.analyze_frame(
                bgr, coin_mm=coin_mm,
                distance_mm=dist_mm,       # None for frames (no EXIF)
                assume_mm=assu_mm, coin_assumed=True)
    except yolo_mode.ModelNotTrained as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Live analysis failed: {exc}"}), 500

    # frame size (after any server-side resize) so the browser canvas
    # can line up the overlay boxes exactly
    rep["frame_w"], rep["frame_h"] = int(bgr.shape[1]), int(bgr.shape[0])
    return jsonify({"ok": True, "rep": clean_report(rep)})


@app.route("/api/mode-info")
def api_mode_info():
    """Tells the page whether YOLO mode is available (trained model?)."""
    ready = yolo_mode.model_ready()
    msg = ("YOLOv8 onion model loaded." if ready else
           "YOLO mode needs a fine-tuned model (models/onion_yolo.pt). "
           "The free pretrained YOLOv8 does not know onions - train it "
           "with train_yolo.py on your labeled photos. Classic CV mode "
           "is fully working meanwhile.")
    return jsonify({"yolo_ready": ready, "message": msg})


@app.route("/outputs/<path:name>")
def outputs(name):
    """Download link for report files (annotated, json, txt, card)."""
    return send_from_directory(OUTPUT_DIR, name, as_attachment=False)


# ------------------------------------------------------------------
# OFFLINE APP (PWA): the whole pipeline runs in the BROWSER via
# OpenCV.js - after the first load it works with no internet at all.
# Serve with this Flask app at  http://localhost:8000/offline/
# ------------------------------------------------------------------
OFFLINE_DIR = os.path.join(BASE_DIR, "offline")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.route("/assets/<path:name>")
def static_files(name):
    """Design assets (hero art etc.). NOTE: Flask reserves /static/ for
    its own handler, so design assets live under /assets/."""
    return send_from_directory(FRONTEND_DIR, name)


@app.route("/offline/")
def offline_index():
    return send_from_directory(OFFLINE_DIR, "index.html")


@app.route("/offline/<path:name>")
def offline_file(name):
    return send_from_directory(OFFLINE_DIR, name)


@app.route("/uploads/<path:name>")
def uploads(name):
    """Serve the uploaded original photo so it can be shown on the page."""
    return send_from_directory(UPLOAD_DIR, name, as_attachment=False)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "onion-quality-grader"})


@app.errorhandler(413)
def upload_too_large(e):
    """Photo bigger than MAX_CONTENT_LENGTH -> friendly JSON, not an HTML
    error page (the page's JavaScript only understands JSON).

    Why the limit: Vercel's serverless functions hard-reject request
    bodies over 4.5 MB, so SERVERLESS mode caps uploads at 4 MB. The
    web page shrinks big phone photos in the browser before sending
    (see shrinkPhoto in app_page.html), so real users never see this."""
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"ok": False,
                    "error": f"Photo too large (max {limit_mb} MB on this "
                             "host). Use a smaller/compressed photo, or the "
                             "LIVE CAMERA mode."}), 413


# ------------------------------------------------------------------
# Start the server
# ------------------------------------------------------------------
def local_ip():
    """Best-effort guess of this laptop's address on the Wi-Fi network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    ip = local_ip()
    print("=" * 62)
    print(" 🧅 ONION QUALITY GRADER - web app is starting...")
    print(f" On this laptop : http://localhost:8000")
    print(f" On your phone  : http://{ip}:8000   (SAME Wi-Fi network!)")
    print(" Honest note    : visible-surface analysis only.")
    print("=" * 62)
    # 0.0.0.0 = accept connections from other devices (the phone)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
