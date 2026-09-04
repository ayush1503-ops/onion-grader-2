# Appendix A — Project Structure & Full Code

## A.1 The structure (Part 14)
```text
onion-quality-analyzer/
├── dataset/                 # your labeled onion photos (train/val/test)
│   ├── train/  good/ damaged/ rotten/
│   ├── val/    good/ damaged/ rotten/
│   └── test/   good/ damaged/ rotten/
├── images/                  # input + demo images, pipeline outputs
├── src/
│   ├── __init__.py
│   └── onion_analyzer.py    # the full pipeline (below)
├── results/                 # saved result images + reports
├── tests/                   # small automated checks
├── main.py                  # command-line entry point
├── app.py                   # optional Streamlit web UI
├── requirements.txt
└── README.md
```

## A.2 `requirements.txt`
```text
opencv-python
numpy
# optional:
# streamlit
# scikit-learn
# pandas
# joblib
```

## A.3 `src/onion_analyzer.py` — the complete analyzer
```python
"""
onion_analyzer.py — rule-based Onion Quality Image Analyzer.
Pipeline: read -> resize -> gray -> blur -> Otsu -> morphology ->
          contour -> mask -> features -> rules -> result.
Every intermediate image is saved for debugging.
Demo thresholds only — tune against a labeled dataset (see book Ch 24).
"""
import os, sys
import cv2
import numpy as np

def read_and_resize(path, width=640):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)),
                         interpolation=cv2.INTER_AREA)
    return img

def segment_onion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour found.")
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return gray, blurred, thresh, opened, largest, mask, masked_img

def extract_features(mask, gray, hsv):
    area = float(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(c, True))
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0.0
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(h) / w if w else 0.0
    extent = area / float(w * h) if w * h else 0.0
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    std_gray = float(gray_masked[np.nonzero(mask)].std())
    dark_ratio = float(np.mean(gray[mask > 0] < 70))
    H, S, V = cv2.split(hsv)
    brown = ((H >= 8) & (H <= 25) & (S >= 40) & (V <= 200)).astype(np.uint8)
    brown_ratio = float(brown[mask > 0].mean())
    return {"area": area, "perimeter": perimeter, "circularity": circularity,
            "aspect_ratio": aspect_ratio, "extent": extent,
            "std_gray": std_gray, "dark_ratio": dark_ratio,
            "brown_ratio": brown_ratio,
            "defect_ratio": max(dark_ratio, brown_ratio)}

def classify(feats):
    dr, br = feats["dark_ratio"], feats["brown_ratio"]
    if dr >= 0.12 or br >= 0.40:
        return "ROTTEN", (0, 0, 255)
    if dr >= 0.04 or br >= 0.15:
        return "DAMAGED", (0, 165, 255)
    return "GOOD", (0, 200, 0)

def annotate(img, mask, label, color, feats):
    result = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 100, 0), 3)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(result, f"{label}  (defect={feats['defect_ratio']:.2f})",
                (x, max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
    return result

def analyze(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = read_and_resize(path)
    gray, blurred, thresh, opened, largest, mask, masked_img = segment_onion(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = extract_features(mask, gray, hsv)
    label, color = classify(feats)
    result = annotate(img, mask, label, color, feats)
    cv2.imwrite(os.path.join(out_dir, "01_original.jpg"), img)
    cv2.imwrite(os.path.join(out_dir, "02_grayscale.jpg"), gray)
    cv2.imwrite(os.path.join(out_dir, "03_blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(out_dir, "04_threshold.jpg"), thresh)
    cv2.imwrite(os.path.join(out_dir, "05_morphology.jpg"), opened)
    cv2.imwrite(os.path.join(out_dir, "07_mask.jpg"), mask)
    cv2.imwrite(os.path.join(out_dir, "08_masked_onion.jpg"), masked_img)
    cv2.imwrite(os.path.join(out_dir, "09_result.jpg"), result)
    return label, feats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onion_analyzer.py <image> [out_dir]")
        sys.exit(1)
    label, feats = analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
    print(f"RESULT: {label}")
    for k, v in feats.items():
        print(f"  {k:14s} = {v}")
    print("NOTE: visible surface analysis only — cannot detect internal defects.")
```

## A.4 `main.py` — the entry point
```python
import sys, cv2
from src.onion_analyzer import analyze

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    label, feats = analyze(sys.argv[1], "results")
    img = cv2.imread("results/09_result.jpg")
    print(f"RESULT: {label}")
    print("NOTE: visible surface analysis only — cannot detect internal defects.")
    cv2.imshow("Onion Quality Analyzer", img)
    cv2.waitKey(0); cv2.destroyAllWindows()
```

## A.5 `app.py` — optional Streamlit web UI (free)
```python
import streamlit as st
from src.onion_analyzer import analyze

st.title("🧅 Onion Quality Analyzer")
st.caption("Grades *visible* surface quality from one photo. "
           "Cannot detect internal defects.")
f = st.file_uploader("Upload an onion photo", type=["jpg", "jpeg", "png"])
if f is not None:
    with open("upload.jpg", "wb") as out:
        out.write(f.getbuffer())
    st.image("upload.jpg", width=300)
    label, feats = analyze("upload.jpg", "results")
    st.subheader(f"Result: {label}")
    st.write({k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()})
    st.image("results/09_result.jpg", width=300)
```
Run: `pip install streamlit && streamlit run app.py`

---

# Appendix B — Your AI Prompt Library (Part 5 & 12)

> Replace `[…]` with your own details. Use these with any free AI chat.

## Understanding OpenCV
- "Explain `[…]` (an OpenCV function) like I'm a beginner. What does each argument mean? Give a tiny working example and a common mistake."
- "What is `[…]` used for in real computer-vision projects? Give 2 real examples."

## Explaining code
- "Explain this code line by line for a beginner: [paste code]"
- "What would happen if I changed `[…]` to `[…]` in this code? [paste code]"

## Debugging errors
- "I'm a beginner. Here is the full error I got: [paste error]. Explain in simple words what caused it, and give me the exact line to change."
- "What are the 5 most common causes of this error: [paste error]? How do I check each one?"

## Generating practice questions
- "Give me 5 beginner practice questions on `[…]` with solutions, so I can test myself."
- "Quiz me on `[…]`: ask one question, wait for my answer, then tell me if I'm right."

## Reviewing my code
- "Review this code for bugs and style. Suggest improvements and explain each one: [paste code]"
- "Is this a good way to `[…]`? What's a better way and why? [paste code]"

## Explaining screenshots
- "I'll describe what I see in my program's output: [describe]. Why does it look like this, and what should I check?"

## Improving my project
- "My onion analyzer misclassifies `[…]`. Given these features [list], what should I change or investigate first?"
- "Suggest 3 experiments to make my classifier more robust on new photos."

## Documentation
- "Write a README for this project: [paste structure + code]. Include setup, usage, and honest limitations."

## Brainstorming experiments
- "List 5 cheap experiments to test whether my features (color, shape, texture) actually separate good vs rotten onions."

## Debugging workflow prompts (Part 12)
- **Python errors**: "Explain this Python traceback line by line and pinpoint the exact line that failed: [paste traceback]"
- **OpenCV errors**: "I got `cv2.error: …`. Which OpenCV call is failing and why? [paste error + code]"
- **Image-loading errors**: "cv2.imread returns None for my file. List the checks I should run (path, permissions, format, case)."
- **Path errors**: "My code can't find `dataset/test/good/…`. Show me how to debug relative vs absolute paths."
- **Segmentation problems**: "My mask includes the background. List 4 causes and fixes for Otsu-based segmentation."
- **Threshold problems**: "My threshold keeps most of the image white. What should I check in my lighting/background?"
- **Contour problems**: "I get hundreds of tiny contours instead of one onion. What should I change?"
- **Model errors**: "My model's validation accuracy is much lower than training. Diagnose overfitting and give fixes."
- **Dataset errors**: "My model learns the background, not the onion. How do I detect and fix label/imaging issues?"

> **Golden rule:** always paste the **full error message**, and *ask why* — don't blindly copy a fix. Read the explanation, then apply it yourself.

---

# Appendix C — Debugging Cheatsheet (Ch 12 workflow)

**The loop:** Error → Understand → Find cause → Ask AI → Apply fix → Test → Understand fix.

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cv2` | not installed / wrong env | `pip install opencv-python` in the same Python |
| `img is None` | bad path / missing file | print path; `os.path.exists`; check case |
| `(-215:Assertion failed) !ssize.empty()` | empty image passed | guard `if img is None` |
| window flashes & closes | missing `waitKey` | `cv2.waitKey(0)` before destroy |
| colors swapped | RGB used, OpenCV is BGR | use BGR tuples |
| `resize` output stretched | forgot aspect ratio | scale h and w equally |
| `IndexError` on channel 3 | channels are 0,1,2 | use 0..2 (B,G,R) |
| empty crop | wrong slice order / reversed coords | `img[y1:y2, x1:x2]` + min/max |
| `findContours` error | input not binary / shape wrong | pass a 1-channel binary image |
| hundreds of contours | noise speckles | morphology open/close first, keep largest |
| mask includes background | background not uniform | consistent background; explicit threshold |
| Otsu picks wrong side | object lighter than bg | `THRESH_BINARY` vs `_INV` |
| `imwrite` returns False | missing folder | `os.makedirs(dir, exist_ok=True)` |
| `medianBlur` error | even kernel | use odd kernel (3,5,7…) |
| video loop freezes | `waitKey(0)` in loop | `waitKey(1)` |
| callback never fires | registered after `waitKey` | register before the loop |
| model 99% train / 60% test | overfitting | more data, augmentation, regularization |
| high accuracy but useless | class imbalance | confusion matrix + precision/recall/F1 |
| "ROTTEN" on a clean photo | dark background | control imaging setup (Ch 21) |

---

# Appendix D — Glossary

- **BGR** — OpenCV's channel order (Blue, Green, Red).
- **Binary image** — image with only two values (0 and 255).
- **Channel** — one layer of a color image (e.g., the Red layer).
- **CNN** — convolutional neural network; learns features + rules from pixels.
- **Contour** — the boundary of a white region in a binary image.
- **Feature** — one number summarizing an aspect of the image (area, color, …).
- **Grayscale** — 1-channel image of brightness only.
- **HSV** — Hue/Saturation/Value color space (human-like color).
- **Mask** — image marking which pixels belong to the object.
- **Morphology** — erosion/dilation operations that clean binary blobs.
- **Otsu** — automatic method to pick the best threshold.
- **Overfitting** — memorizing training data; failing on new data.
- **Pixel** — smallest element of an image.
- **Precision / Recall / F1** — per-class metrics beyond accuracy.
- **ROI** — region of interest (a crop).
- **Segmentation** — separating the object from the background.
- **Supervised learning** — learning from labeled examples.
- **Thresholding** — turning grayscale into black/white via a cutoff.
- **Transfer learning** — reusing a pre-trained network for a new task.
- **uint8** — 8-bit unsigned integer, values 0–255.

---

# Appendix E — Answer Key (all end-of-chapter blocks)

All answers are inline at the end of every chapter (see the **✅ Answers** lines). The Quick Revision answers are compressed there; the full explanations live in each chapter's "Questions + Solutions" section.

---

# 📚 What to do next (your 30-day plan)

1. **Days 1–3** — Chapters 1–2: install everything, read images, understand pixels.
2. **Days 4–7** — Chapters 3–10: the whole playlist (read, gray, channels, resize, flip, crop, save, draw).
3. **Days 8–9** — Chapters 11–13: events + cropping tool + video.
4. **Days 10–13** — Chapters 14–18: HSV, threshold, blur, morphology, contours (the onion toolkit).
5. **Days 14–15** — Chapters 19–20: features + **build the rule-based analyzer**.
6. **Days 16–19** — Chapter 21: photograph & label your dataset (start early!).
7. **Days 20–23** — Chapter 22: train the ML classifier; Chapter 24: evaluate honestly.
8. **Days 24–27** — Chapter 25: finalize the project + UI + README.
9. **Days 28–30** — Chapter 26: write limitations; try a CNN (Ch 23) in Colab if data allows.

**Keep the loop going:** Understand → Plan → Ask AI → Generate → Run → Test → Debug → Understand → Improve. 🧅
