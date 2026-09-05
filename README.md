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

## YOLOv8 detection engine (OpenCV + YOLOv8 combo)

`yolo_mode.py` is the deep-learning engine: **YOLOv8n** finds and boxes
every onion, then the **OpenCV** pipeline (`grader.py`) takes over for the
coin-scale (mm), defect classes, grading and reports — the same honest
logic as classic CV mode. In the web app tick **“YOLOv8 engine”** in the
toolbar; live camera YOLO works too.

The shipped `models/onion_yolo.pt` is fine-tuned on the SYNTHETIC dummy
dataset (`make_dummy_detection_dataset.py` → `train_yolo.py`), mAP50 ≈ 0.99
on the synthetic val split (demo numbers only). CLI:

```bash
python yolo_mode.py dataset_yolo/demo/demo_1.jpg --classifier yolo
```

Retrain on real photos: label them (Roboflow/LabelImg/CVAT), then
`python train_yolo.py` — use `--scratch` to train fully offline (no
pretrained weights download), or `--model yolov8n.yaml` for the same.

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
- YOLO mode is **active**: `models/onion_yolo.pt` ships a YOLOv8n detector
  fine-tuned on the SYNTHETIC dummy dataset (`make_dummy_detection_dataset.py`).
  It reaches **mAP50 ≈ 0.99 on that synthetic validation set** — honest demo
  numbers only, it says nothing about real photos. Toggle **YOLOv8 engine**
  in the web app. To grade real photos accurately, retrain on labeled real
  data: `python train_yolo.py` (or `--scratch` to train fully offline).

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
