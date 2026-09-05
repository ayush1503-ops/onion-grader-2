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
from flask import Flask, jsonify, request, send_from_directory, make_response

import grader
import onion_presence
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
# Extensions we accept. HEIC/HEIF are what iPhones produce by default:
# the browser converts them to JPEG before upload (see toUploadJpeg in
# app_page.html), but the file NAME often still ends in .heic, so the
# name alone must never be the reason an upload is refused.
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
               ".heic", ".heif", ".jfif", ".gif", ".tif", ".tiff"}
# STEP 0 gate: the scikit-learn "is there an onion here?" model
# (onion_presence.py). Set ONION_PRESENCE=0 to switch it off.
PRESENCE_ENABLED = os.environ.get("ONION_PRESENCE", "1") != "0"
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
# Helper: decode ANY photo a phone might send
# ------------------------------------------------------------------
def _exif_rotate(bgr, raw):
    """Apply the EXIF orientation flag phones set instead of rotating.

    A photo taken in portrait on a phone is usually stored LANDSCAPE with
    an "orientation: 6" tag. OpenCV ignores that tag, so the pipeline saw
    sideways photos - onions came out measured along the wrong axis and
    the annotated report looked rotated. Pillow reads the tag for us.
    """
    try:
        from PIL import Image as PILImage
        import io
        pil = PILImage.open(io.BytesIO(raw))
        orient = (pil.getexif() or {}).get(274, 1)     # 274 = Orientation
    except Exception:
        return bgr
    if orient == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if orient == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if orient == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orient == 2:
        return cv2.flip(bgr, 1)
    if orient == 4:
        return cv2.flip(bgr, 0)
    if orient == 5:
        return cv2.rotate(cv2.flip(bgr, 1), cv2.ROTATE_90_CLOCKWISE)
    if orient == 7:
        return cv2.rotate(cv2.flip(bgr, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def decode_photo(raw):
    """bytes -> upright BGR image, or None if it is not a readable photo.

    Tries, in order:
      1. OpenCV        - JPEG / PNG / WEBP / BMP / TIFF
      2. Pillow        - anything else OpenCV skips, incl. HEIC/HEIF when
                         pillow-heif is installed (optional dependency)
    Then applies the EXIF orientation so portrait phone photos are upright.
    """
    if not raw:
        return None
    buf = np.frombuffer(raw, np.uint8)
    # IMREAD_IGNORE_ORIENTATION: OpenCV >= 4.7 silently applies the EXIF
    # rotation itself, older builds do not. We turn its version-dependent
    # behaviour OFF and always rotate ourselves, so the result is
    # identical on every host (double-rotation was landing phone photos
    # sideways on new OpenCV builds).
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        try:
            import io
            from PIL import Image as PILImage
            try:                       # optional: iPhone HEIC support
                import pillow_heif
                pillow_heif.register_heif_opener()
            except Exception:
                pass
            pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if bgr is None or bgr.size == 0:
        return None
    return _exif_rotate(bgr, raw)


# ------------------------------------------------------------------
# Helper: run the whole pipeline on one upload
# ------------------------------------------------------------------
def run_pipeline(file_storage, coin_preset, coin_custom, batch_id, mode="cv",
                 distance_cm=None, assume_mm=None):
    """Save the upload -> run the pipeline -> build download URLs."""
    # ---- MOBILE-SAFE INTAKE -------------------------------------
    # Phone browsers are wildly inconsistent about what they send:
    #   * iOS Safari camera captures often arrive as "image.jpg" but can
    #     also be HEIC, or carry NO filename at all (blob uploads).
    #   * Some Android WebViews send "blob" or an empty filename.
    #   * Photos are routinely rotated only by an EXIF orientation flag.
    # So we decode the BYTES and never trust the file name. The name is
    # used for the saved copy only, and we always normalise it to .jpg.
    raw = file_storage.read()
    if not raw:
        return None, ("The photo did not arrive (0 bytes). This usually "
                      "means the upload was interrupted - try again, or "
                      "pick the photo from your gallery instead of the "
                      "camera.")

    filename = secure_filename(file_storage.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in ALLOWED_EXT:
        return None, ("That file is not a photo. Please choose a JPG, PNG, "
                      "WEBP or HEIC image.")

    bgr = decode_photo(raw)
    if bgr is None:
        return None, ("Could not read that photo. If it came from an "
                      "iPhone it may be in HEIC format - open it once in "
                      "the Photos app and share/save it as JPEG, or set "
                      "Settings > Camera > Formats > Most Compatible.")

    # Re-encode to a plain JPEG: one predictable format on disk, EXIF
    # rotation already applied, and the report images stay correct.
    stem = os.path.splitext(os.path.basename(filename))[0] or "photo"
    stamp = grader.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_name = f"{stamp}_{stem}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    ok_write = cv2.imwrite(save_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok_write:
        return None, "Could not save the uploaded photo on the server."

    coin_mm, dist_mm, assu_mm, err = resolve_coin(
        coin_preset, coin_custom, distance_cm, assume_mm)
    if err:
        return None, err
    batch_id = (batch_id or "").strip() or None   # empty -> auto batch id

    # ---- STEP 0: is there an onion at all? (scikit-learn model) ----
    # Runs BEFORE grading so a photo of a tomato / an empty table gets
    # an honest "Onion not found" instead of invented grades.
    presence = None
    if PRESENCE_ENABLED:
        presence = onion_presence.check(bgr)
        if not presence["is_onion"]:
            return None, (onion_presence.NOT_FOUND_MSG + " Reason: "
                          + presence["reason"])

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

    # Detection failure is an error, not a 0% grade of an empty batch.
    if not rep.get("onion_detected", rep.get("onion_count", 0) > 0):
        return None, grader.NO_ONION_ERROR

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
    out = {"rep": clean_report(rep), "urls": urls}
    if presence:
        out["presence"] = presence
        out["rep"]["presence"] = presence
    return out, None


def file_urls(rep):
    """Download links for the report files of an analysis.

    Serverless mode (VERCEL=1): the disk is ephemeral, so files are
    embedded straight into the JSON response as data: URIs instead of
    links that would break on the next request.

    Every response also carries urls["names"][key] = the suggested file
    name, so the page's download buttons (Blob-based, see dlKey in
    app_page.html) can save data: URIs with a proper name too - a plain
    <a download> link FAILS for data: URIs in Firefox/Safari and is
    ignored by mobile browsers, which is why "report is not downloading"
    happened on some phones/hosts.
    """
    names = [os.path.basename(f) for f in rep.get("files", [])]
    if len(names) < 4:
        return {}
    keys = ["annotated", "json", "txt", "card", "full"][:len(names)]
    if os.environ.get("VERCEL") == "1":
        import base64 as _b64
        mimes = {0: "image/jpeg", 1: "application/json",
                 2: "text/plain", 3: "image/jpeg", 4: "image/jpeg"}
        urls = {}
        for i, key in enumerate(keys):
            path = os.path.join(OUTPUT_DIR, names[i])
            with open(path, "rb") as fh:
                urls[key] = (f"data:{mimes[i]};base64,"
                             + _b64.b64encode(fh.read()).decode())
        urls["names"] = {key: names[i] for i, key in enumerate(keys)}
        return urls
    urls = {key: f"/outputs/{names[i]}" for i, key in enumerate(keys)}
    urls["names"] = {key: names[i] for i, key in enumerate(keys)}
    return urls


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    """The one and only page (form + JavaScript that renders results).
    no-cache: the HTML is small and edits are frequent - browsers must
    always revalidate so users never see a stale design after a deploy."""
    with open(os.path.join(BASE_DIR, "app_page.html"), "r",
              encoding="utf-8") as fh:
        page = fh.read()
    resp = make_response(page)
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """The page sends the photo here. Returns JSON results."""
    # Accept a few field names: "photo" is ours, but integrations and
    # some mobile share-targets post "file" / "image" / "frame".
    photo = (request.files.get("photo") or request.files.get("file")
             or request.files.get("image") or request.files.get("frame"))
    if photo is None and request.files:
        # last resort: take whatever single file was posted. iOS Safari
        # sometimes sends a blob under an empty/odd field name, and
        # rejecting it was the "Analyse does nothing on iPhone" bug.
        photo = next(iter(request.files.values()), None)
    if photo is None:
        return jsonify({"ok": False,
                        "error": "Please choose a photo first."}), 400
    # NOTE: deliberately NOT checking photo.filename here - blob uploads
    # legitimately have no name. run_pipeline() validates the BYTES.
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
    bgr = decode_photo(data)
    if bgr is None:
        return jsonify({"ok": False,
                        "error": "Frame was not a valid image."}), 400
    # normalize to the LIVE working width FIRST, so the overlay boxes
    # (drawn in the analyzed frame's coordinates) line up exactly with
    # the canvas size we report below
    bgr = grader.fit_live_frame(bgr)

    # STEP 0 (live): the scikit-learn presence model. A frame with no
    # onion returns HTTP 200 + no_onion so scanning keeps running.
    presence = onion_presence.check(bgr) if PRESENCE_ENABLED else None
    if presence and not presence["is_onion"]:
        return jsonify({"ok": True, "no_onion": True, "presence": presence,
                        "rep": {"onion_count": 0, "onions": [],
                                "frame_w": int(bgr.shape[1]),
                                "frame_h": int(bgr.shape[0])},
                        "error": onion_presence.NOT_FOUND_MSG})

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
    payload = {"ok": True, "rep": clean_report(rep)}
    if presence:
        payload["presence"] = presence
    if not rep.get("onion_detected", rep.get("onion_count", 0) > 0):
        # Live scanning must keep running (HTTP 200) so the next frame
        # can recover; the page shows this as a red "no onion" error.
        payload["no_onion"] = True
        payload["error"] = grader.NO_ONION_LIVE
    return jsonify(payload)


@app.route("/api/mode-info")
def api_mode_info():
    """Tells the page whether YOLO mode is available (trained model?)."""
    ready = yolo_mode.model_ready()
    msg = ("YOLOv8 onion model loaded." if ready else
           "YOLO mode needs a fine-tuned model (models/onion_yolo.pt). "
           "The free pretrained YOLOv8 does not know onions - train it "
           "with train_yolo.py on your labeled photos. Classic CV mode "
           "is fully working meanwhile.")
    return jsonify({"yolo_ready": ready, "message": msg,
                    "presence": onion_presence.info(),
                    "presence_enabled": PRESENCE_ENABLED})


@app.route("/api/detect-onion", methods=["POST"])
def api_detect_onion():
    """Onion / not-onion only (scikit-learn), no grading.

    Handy for integrations: POST a photo as "photo" (or "frame") and get
    {"ok":true,"is_onion":false,"message":"Onion not found ..."}"""
    up = request.files.get("photo") or request.files.get("frame")
    if up is None:
        return jsonify({"ok": False, "error": "Send a photo file."}), 400
    bgr = decode_photo(up.read())
    if bgr is None:
        return jsonify({"ok": False, "error": "Not a readable image."}), 400
    v = onion_presence.check(bgr)
    return jsonify({"ok": True, **v, "model_info": onion_presence.info(),
                    "message": ("Onion detected." if v["is_onion"]
                                else onion_presence.NOT_FOUND_MSG)})


def _safe_name(name):
    """Reject path tricks (../, absolute paths) - files live flat in one dir."""
    base = os.path.basename(name or "")
    if not base or base != name or ".." in base or base.startswith("."):
        return None
    return base


@app.route("/outputs/<path:name>")
def outputs(name):
    """Serve report files (annotated, json, txt, card, full).

    Default = inline (so <img> previews keep working). Add ?download=1
    to force a "save file" dialog even in browsers that ignore the
    <a download> attribute (mobile Safari and friends)."""
    safe = _safe_name(name)
    if safe is None:
        return jsonify({"ok": False, "error": "bad file name"}), 400
    as_dl = request.args.get("download") == "1"
    return send_from_directory(OUTPUT_DIR, safe, as_attachment=as_dl,
                               download_name=safe)


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
    safe = _safe_name(name)
    if safe is None:
        return jsonify({"ok": False, "error": "bad file name"}), 400
    as_dl = request.args.get("download") == "1"
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=as_dl,
                               download_name=safe)


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
