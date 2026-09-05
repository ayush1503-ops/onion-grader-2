# 🧅 OnionGrader — AI Onion Quality Grading (SIH26031)

Grade onion batches from a single photo: **count, size in millimetres,
quality classes, layer analysis, weight, reports** — plus **live camera
scanning**, a **fully offline phone app**, **CNN training pipelines**
(PyTorch + TensorFlow + transfer learning), a **batch dashboard** and an
**integration API** (JSON/CSV/PDF).

> **Honest by design** (read this first):
> - Grades the **visible surface only**. No photo app can detect internal
>   rot, internal damage or moisture — we never claim otherwise.
> - Sizes are only as good as the scale reference: a **coin in frame =
>   exact mm**, camera-distance = ±20%, otherwise clearly-labeled
>   assumptions.
> - Every accuracy number in this repo is labeled **SYNTHETIC demo data**.
>   Real accuracy requires a real labeled test set. We never invent numbers.

---

## Quick start (2 minutes)

```bash
pip install -r requirements.txt        # core = Flask, OpenCV, Pillow, numpy
python app.py                          # → http://localhost:8000
```

Open the printed link on your phone (same Wi-Fi) for the live camera.
Click the **हिं** button in the nav for the Hindi UI.

| What | Where |
|---|---|
| Main app (upload + live scan) | `python app.py` → :8000 |
| "Is it an onion?" check (scikit-learn) | `POST /api/detect-onion` |
| Offline phone app (PWA, on-device AI) | `/offline/` → Add to Home Screen |
| Batch dashboard | `python dashboard.py` → :8002 |
| Integration API (JSON/CSV/PDF) | `python api.py` → :8001 |
| Production server | `python wsgi.py` (waitress) |

## How the grading works (classic CV pipeline)

`grader.py`: background-robust Otsu mask → blob cleanup → **watershed +
Hough splitting** for touching/piled onions → coin detection for mm scale
(auto: ₹10/₹2 = 27 mm) → per-onion defect classification (GOOD / DAMAGED /
ROTTEN / SPROUTED / UNDERSIZED) → A / URS / REJECT grades → **layer
analysis** (L1 fully visible / L2 / L3 hidden) → weight estimate
(mass = k·d³) → a **secondary "detect ALL onions" engine** (local-contrast
retry + adaptive + Hough sweep) rescues low-contrast/uneven-light onions.
Every analysis produces: `annotated.jpg`, `report.json` (embeds the photo),
`report.txt`, `report_card.jpg`, and `full_report.jpg` (the whole report
as ONE image).

## Step 0 — "Is there an onion at all?" (scikit-learn)

Before anything is graded, `onion_presence.py` asks one question about
the whole photo: **does this picture contain onions?** If not, the app
answers honestly — **"Onion not found in this image."** — instead of
inventing grades for a tomato, an apple or an empty table.

* **Model** — a scikit-learn `RandomForestClassifier` (250 trees) over
  23 OpenCV features: blob count/roundness/solidity, HSV + Lab colour
  statistics, palette fractions (green / blue / purple / gloss / vivid /
  tomato-red), Laplacian micro-texture and Canny edge density —
  i.e. the papery, matte, low-gloss look of onion skin.
* **Runtime is numpy-only** — the forest is exported to
  `models/onion_presence.json` (like `models/onion_clf.json`), so the
  web app and the Vercel deployment need **no scikit-learn** to predict.
* **Wired into everything** — upload (`/api/analyze`), live camera
  (`/api/live`), YOLO mode (`yolo_mode.analyze`) and a standalone
  endpoint `POST /api/detect-onion`. The web page renders a dedicated
  "Onion not found" card with the measured reason, and a
  `onion check 99% (sklearn)` chip on every real report.

```bash
python train_presence.py            # re-train + export the JSON model
python selftest_presence.py         # regression check (45/45 passing)
python onion_presence.py photo.jpg  # CLI: ONION / NOT FOUND + reason
curl -F photo=@photo.jpg http://localhost:8000/api/detect-onion
```

**Honest numbers**: grouped 5-fold cross-validation (a photo and its
augmented copies never straddle the split) = **0.83 accuracy** on this
repo's 70-odd source images — real onion photos, tomato / potato /
apple / empty-table negatives, plus synthetic scenes. That is a
small-sample number on this data, not a claim about all photos. Drop
your own photos in `dataset_presence/positive` and
`dataset_presence/negative` and re-run `train_presence.py` to improve
it.

## YOLOv8 engine (OpenCV + YOLOv8 combo)

`models/onion_yolo.pt` ships fine-tuned and the app auto-enables **YOLO
mode** when it is present: YOLOv8 boxes every onion, then the OpenCV
pipeline (`grader.py`) measures mm, classifies the surface and grades
the batch — detector from deep learning, measurement from classic CV.

```bash
python make_dummy_detection_dataset.py     # scenes + labels
python train_yolo.py --epochs 40 --imgsz 416 --batch 16
python yolo_mode.py photo.jpg --classifier rules
```

The shipped weights were trained **on this repo's synthetic detection
scenes** (220 train / 40 val, 40 epochs, from scratch — the sandbox had
no internet for the COCO-pretrained checkpoint). Measured on its own
synthetic val split: **mAP50 0.995, mAP50-95 0.954, P 0.999, R 0.994**.
Those are SYNTHETIC numbers — they prove the pipeline, not field
accuracy. For real deployment, fine-tune on labeled real photos
(Colab T4, see `train_yolo.py`) and drop the new `best.pt` in as
`models/onion_yolo.pt`.

## Detect only onions (the ONION-ONLY gate)

The segmenter finds *any* blob that stands out from the background, so an
apple, a potato, a hand or a green shoot used to be counted and graded as
an onion. `grader.verify_onions()` now asks the question the pipeline
never asked — **"does this blob LOOK like onion skin?"** — in two ways:

1. **Surface palette** — rejects surfaces onion skin never has: vivid
   green (leaves/shoots), blue/cyan (plastic, cloth), purple (brinjal),
   mirror-gloss (glossy fruit/plastic), plain smooth white (lids, cups).
2. **Photo consistency** — inside one photo every onion shares variety
   and light, so a blob that is far brighter/darker, far more colourful
   or a different hue than *every* other onion in the same photo is a
   foreign object (one apple in a pile of red onions).

Rejected objects are **disclosed, never silently dropped**: the report
carries `rejected_not_onion` + `not_onion_reasons`, the web UI shows a
"not onion skin — n rejected" chip, and the app warns when a photo holds
no onion-looking object at all.

Measure it on your own photos (nothing is downloaded):

```bash
mkdir -p dataset_gate/onion dataset_gate/not_onion
#   drop real onion photos in the first folder, apples/potatoes/hands/...
#   in the second one
python evaluate_gate.py                 # gate ON vs OFF, same photos
python evaluate_gate.py --mixed         # also paste each foreign object
                                        # into each onion photo
```

**Honest numbers** (measured 2026-09 on 19 onion photos + 32 non-onion
photos): **94.5 % of real onion blobs kept**, **45.6 % of non-onion
objects rejected**, **100 %** of the synthetic look-alikes in
`test_batch_8_not_onions.jpg`. A lone potato, yellow apple, lemon or
garlic bulb is made of the same browns and yellows as an onion — colour
alone cannot separate those, and pushing the thresholds until they are
caught starts deleting real onions. For true onion-only detection train a
detector on labeled photos (it learns the *shape*, not just the colour):
`prelabel_real.py` → `train_yolo.py` → `models/onion_yolo.pt`, then use
YOLO mode in the app.

## Live camera system

Quality meter (blur/brightness gates bad frames), torch + zoom (where the
phone supports it), alignment guides, scene-change detection, best-frame
memory, and median-of-3 smoothing so live tallies don't flicker.

## CNN training (PyTorch + TensorFlow)

```bash
python make_dataset.py         # 800 SYNTHETIC labeled onions (swap real ones in)
python pytorch_cnn.py          # custom CNN        → test 0.767 (synthetic!)
python tensorflow_cnn.py       # same idea, Keras  → test 0.917 (synthetic!)
python transfer_learning.py    # MobileNetV2       → test 1.000 (synthetic!)
python evaluate.py             # confusion matrices, TEST set only, honest
python export_models.py        # ONNX + TFLite, both verified
```

The measured lesson: **transfer learning beats custom CNNs on <1000
images** (pretrained features > learned-from-scratch). See
`COLAB_TRAINING.md` for free-T4 GPU training and how to swap in real
photos — same folder layout, zero code changes.

## Build a REAL dataset fast

```bash
python sam_label.py my_photos/         # detector proposes boxes,
                                       # YOU press 1 key per crop:
                                       # 1=good 2=sprouted 3=rotten 4=cut
```

## Conveyor belt counting

```bash
python tracker.py belt_video.mp4       # unique-ID tracking, counts each
python tracker.py 0                    # onion ONCE (webcam works too)
```

## File map

```
app.py  app_page.html      main web app (Flask) + UI
grader.py                  the whole CV pipeline + report generators
camera.py                  CLI live camera (laptop webcam)
offline/                   PWA: on-device AI, works with NO internet
make_test_images.py        8 synthetic test batches (regression set,
                           batch 8 = NOT onions, for the onion-only gate)
evaluate_gate.py           measures the onion-only gate on your photos
pytorch_cnn.py  tensorflow_cnn.py  transfer_learning.py
make_dataset.py  evaluate.py  export_models.py  COLAB_TRAINING.md
sam_label.py  tracker.py   dataset labeling + belt counting
dashboard.py  api.py       multi-batch view + integration API
wsgi.py  Procfile  DEPLOY.md  .gitignore   production deployment
models/  dataset/  outputs/   trained models, data, saved reports
```

## Tests / verification

```bash
python make_test_images.py                    # rebuild the 8 test batches
python grader.py test_images/test_batch_1.jpg # CLI, prints the report
python selftest.py                            # all batches + gate: exit 0
python evaluate_gate.py                       # onion-only gate, real photos
python tracker.py /tmp/belt.mp4 --no-gui      # expect: UNIQUE ONIONS = 2
python evaluate.py                            # confusion matrices
```

Regression counts (all must hold): 8 / 6 / 4 / 6 / 8 / 4 / 4 onions and
**0 onions** in `test_batch_8_not_onions.jpg` (2 green balls, 1 blue cup,
1 purple brinjal, 1 glossy sphere — all must be rejected).

## Deployment

See `DEPLOY.md` — from a one-command public demo link (Cloudflare
Tunnel) to Render.com, PWA install, and Play-Store APK via PWABuilder.

## Roadmap / honest limits

- Internal quality (rot, moisture, damage inside) is **impossible from a
  normal photo** — needs NIR/X-ray/CT. Out of scope, stated everywhere.
- Real-world accuracy: pending a labeled photo dataset (use
  `sam_label.py` + `COLAB_TRAINING.md`).
- YOLO mode exists (`yolo_mode.py`) but is honest-gated: it refuses to
  fake results without a trained model file.

---

## 🚀 Deploying (GitHub Pages / Render / Vercel)

**[DEPLOY.md](DEPLOY.md)** has every level, from a Wi-Fi demo to a Play
Store APK. The short version:

| What | Where | How |
|---|---|---|
| Offline phone app (`offline/`) | **GitHub Pages** (free HTTPS) | repo *Settings → Pages → Source: GitHub Actions*; the bundled `deploy/github-pages.yml` workflow publishes `offline/` on every push to `main` (move it to `.github/workflows/` once — GitHub blocks bots from adding workflow files) |
| Server app (`app.py`, live camera, reports) | **Render** or **Vercel** | `render.yaml` / `vercel.json` are already in this repo — import the repo and deploy |

## ✅ Detection self-test

```bash
python make_test_images.py   # regenerate the 7 synthetic batches (prints ground truth)
python selftest.py           # checks count / classes / scale source / mm per batch
```

The selftest passes on all 7 synthetic batches (mixed, touching, no
coin, dark tray, pile, EXIF distance, uneven light). All accuracy
claims remain labeled SYNTHETIC — real numbers need a real labeled set.
