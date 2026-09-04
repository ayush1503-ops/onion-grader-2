# 🧅 ONION QUALITY ANALYZER
## The Complete OpenCV + Computer Vision Course Book
### From a beginner playlist to a real working project — with an AI tutor

> **Based on the playlist:** [Image Processing Using OpenCV — CampusX](https://www.youtube.com/playlist?list=PLKnIA16_RmvYXDBJ5WRDuQRSzFJs93pYR)
> **13 videos · ~73 minutes · by CampusX (Nitish Singh) · last updated 20 Mar 2021**
>
> **Your project:** an **Onion Image Quality Analyzer** that reads a photo of an onion and decides whether it looks GOOD, DAMAGED, or ROTTEN.

---

# READ THIS FIRST — how the book works

You asked for a book that (1) turns the entire playlist into a textbook, (2) teaches OpenCV from absolute zero, (3) uses free AI tools everywhere, (4) builds your onion project step by step, and (5) is **highly visual with practical screenshots**.

This book does all five. Three labels tell you exactly where every idea comes from:

| Label | Meaning |
|-------|---------|
| 🎥 **PLAYLIST CONTENT** | This topic **is** in one of the 13 playlist videos. |
| ➕ **ADDITIONAL PROJECT KNOWLEDGE** | This topic is **not** in the playlist. I added it because your onion project needs it. |
| 🧅 **ONION PROJECT** | How the idea applies to your onion analyzer. |

And two labels tell you what kind of image you are looking at:

| Label | Meaning |
|-------|---------|
| 📸 **Real screenshot** | Produced by **actually running the code** (in this environment or by you). |
| 🖼️ **Illustrative screenshot** | Recreated/illustrated. I **never** pretend an illustration is a real screenshot. |

## ⚠️ Honesty rules I promise to follow

1. I analyzed the real playlist page and list **every** video — nothing is skipped, nothing is invented.
2. The playlist covers **beginner basics only** (read, resize, flip, crop, save, draw, events, video). The advanced topics your project needs (HSV, thresholding, contours, ML, CNN…) are added and **clearly labeled** as additional knowledge.
3. I do **not** claim the instructor said things I can't verify. I describe each video's code at the level its title describes, and I tell you what's inferred.
4. I do **not** claim a normal camera can see inside an onion. Chapter 26 says exactly what is and isn't scientifically reasonable.

## What you will build

By the end of this book you will have:

- A working **rule-based onion analyzer** (traditional computer vision) — Chapter 20.
- A **dataset** you can photograph and label yourself — Chapter 21.
- A **machine-learning classifier** that improves on the rules — Chapter 22.
- A simple **upload → analyze → result** interface — Chapter 25.
- A test/evaluation routine that reports **honest** numbers — Chapter 24.

---

# PART A — THE PLAYLIST, FULLY ANALYZED

## A.1 — Every video (nothing skipped)

| # | Video (title) | Length | Video ID | Concepts & tools |
|---|---------------|--------|----------|------------------|
| 1 | Part 1 — Image Basics | 8:02 | `oUJs03eZ0S8` | pixels, grid, dimensions, channels, BGR |
| 2 | Part 2 — Reading an Image | 3:00 | `wRtAoZF50Jc` | `cv2.imread`, `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows` |
| 3 | Part 3 — RGB → Grayscale | 2:00 | `AFrZ3JOQ0Qg` | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| 4 | Part 4 — Color Channels | 3:45 | `wlH9w1eA6PQ` | split / merge B,G,R channels |
| 5 | Part 5 — Image Resizing | 3:27 | `DPkpI2ezVO4` | `cv2.resize`, `INTER_AREA`, `INTER_LINEAR` |
| 6 | Part 6 — Flipping | 2:41 | `Y_78ARbpSwo` | `cv2.flip` (horizontal / vertical / both) |
| 7 | Part 7 — Cropping | 3:17 | `fanEPKLRbPk` | NumPy slicing `img[y1:y2, x1:x2]` |
| 8 | Part 8 — Saving | 1:05 | `b_vVNCVDrbw` | `cv2.imwrite` |
| 9 | Part 9 — Drawing Shapes & Text | 8:49 | `shfXj_Og7ak` | `cv2.line/rectangle/circle/putText` |
| 10 | Part 10 — OpenCV Events | 10:06 | `Vy3SnnSQogA` | mouse events, callback functions |
| 11 | Part 11 — Events Part 2 | 4:53 | `Vw4LA54xG5A` | more mouse events, coordinate reading |
| 12 | Part 12 — Cropping Tool | 10:00 | `_C36hYMa8QU` | combine events + slicing → a real mini-app |
| 13 | Part 13 — Working with Videos | 11:39 | `sg7MwuLsnjQ` | `cv2.VideoCapture`, frame loop, writing video |

> **Honesty note:** I verified the 13 titles, durations and IDs from the playlist page, and I downloaded the videos to extract **real frames** (used as screenshots in this book). I have not transcribed the instructor's spoken words (the audio is Hindi; automatic captions were rate-limited). So I describe each video's code at the level its title describes, using standard OpenCV for that topic. Where I show code it is correct and runnable — I just don't claim "the instructor typed exactly this line."

## A.2 — Video → Concept → Practical → Onion application (the mapping)

| Video | Concept | Practical skill | 🧅 Onion application |
|-------|---------|-----------------|----------------------|
| 1 | pixels, channels, BGR | understand what an image *is* | an onion photo is a 3-channel BGR grid |
| 2 | load a file | `imread` + display | load the onion photo into the analyzer |
| 3 | 3 channels → 1 | `cvtColor` | simpler data = less noise for analysis |
| 4 | isolate B/G/R | split / merge | red channel can highlight dark/brown rot |
| 5 | change size | `resize` | standardise all onion photos |
| 6 | mirror image | `flip` | **data augmentation** — double your dataset |
| 7 | cut a region | NumPy slicing | zoom into the onion, drop the background |
| 8 | write to disk | `imwrite` | save processed / annotated images |
| 9 | annotate | `line/rectangle/circle/putText` | draw "GOOD ✅" and boxes on the onion |
| 10–11 | react to mouse | callbacks | build a click-to-label defect tool |
| 12 | combine skills | mini-app | crop onions out of photos for the dataset |
| 13 | frame loop | `VideoCapture` | analyze onions from a webcam / video |

**Playlist verdict:** it teaches you to *open, view and manipulate* an image. It does **not** teach you to *measure quality*. That is exactly what the additional chapters add.

---

# PART B — TABLE OF CONTENTS

**Phase 1 — Beginner: Python & the playlist (Ch 1–13)**
1. Setting up Python, OpenCV & free tools
2. What is an image? Pixels, resolution, BGR
3. Reading an image
4. Grayscale
5. Color channels
6. Resizing
7. Flipping
8. Cropping
9. Saving images
10. Drawing shapes & text
11. Mouse events
12. Mini-project: the cropping tool
13. Working with video

**Phase 2 — Intermediate: the toolkit your onion needs (Ch 14–20)**
14. Color spaces: HSV & LAB
15. Thresholding & binary images
16. Blurring & noise removal
17. Edge detection & morphology
18. Contours, bounding boxes & masks
19. Histograms & feature extraction
20. Mini-project: the rule-based onion analyzer v1

**Phase 3 — Advanced: from rules to AI (Ch 21–26)**
21. Building your onion dataset
22. Machine learning on features
23. Deep learning / CNN overview
24. Testing & evaluation metrics
25. Final project: full analyzer + simple UI
26. Honest limitations

**Appendices:** project structure & full code · AI prompt library · debugging cheatsheet · glossary · answer key

---

# PART C — FINAL LEARNING ROADMAP

```text
Python Basics                 ← Ch 1–2
      ↓
OpenCV Basics                 ← Ch 3–10
      ↓
Image Processing              ← Ch 14–17
      ↓
Segmentation                  ← Ch 18
      ↓
Feature Extraction            ← Ch 19
      ↓
Traditional Computer Vision   ← Ch 20  ★ FIRST WORKING PROTOTYPE
      ↓
Machine Learning              ← Ch 22
      ↓
Deep Learning (if needed)     ← Ch 23
      ↓
Onion Dataset                 ← Ch 21 (build it in parallel, start early!)
      ↓
Onion Quality Analyzer        ← Ch 20 & 25
      ↓
Testing                       ← Ch 24
      ↓
Final Project                 ← Ch 25
```

**One line per stage:** Python basics = variables, functions, imports, running a script. OpenCV basics = load/show/save/transform without errors. Image processing = make the *useful signal* (quality) stand out. Segmentation = separate "onion" from "not onion". Feature extraction = turn the onion into numbers. Traditional CV = rules over numbers → GOOD/BAD (build first!). Machine learning = let a model learn the rules from labeled examples. Deep learning = a CNN learns features + rules together. Dataset → analyzer → testing → final project.

---

# PART D — YOUR FREE TOOL STACK (verified Aug 2026)

| Purpose | Recommended free tool | Why | Free-tier limit (verified) |
|---------|----------------------|-----|---------------------------|
| Python | [python.org](https://www.python.org) | the language | free forever |
| Computer vision | OpenCV (`opencv-python`) | image processing | free & open-source (BSD) |
| Editor | VS Code | best free editor | free |
| Notebook (cloud) | Google Colab | Jupyter in browser, no install | ~15–30 GPU hrs/wk (dynamic), 12-hr sessions, GPU not guaranteed |
| Notebook (cloud) | Kaggle Notebooks | free GPU + datasets | ~30 GPU hrs/wk, 12-hr sessions |
| Notebook (local) | Jupyter | runs on your PC | free |
| AI coding help | ChatGPT free / Gemini free / Claude free | explain, debug, generate | free tiers exist (message caps vary) |
| Dataset labeling | Roboflow public plan | label + manage images online | free for **public** projects, ~1,000 source images; private = paid |
| Dataset labeling (local) | LabelImg / CVAT | free & open-source | free, runs on your PC |
| ML library | scikit-learn | classic classifiers | free & open-source |
| Deep learning | PyTorch / TensorFlow | CNNs | free & open-source |
| Version control | Git + GitHub | save & share code | free (public repos) |

> Verified Aug 2026: Colab free tier gives roughly 15–30 GPU hours/week dynamically (no fixed published quota, GPU not guaranteed, ~12-hour sessions); Kaggle gives ~30 GPU hours/week (P100/T4); Roboflow's free "public plan" is limited to public projects with a small credit allowance (about 1,000 source images), and private projects/extra training are paid. **I never describe a paid feature as free.**

---

# PART E — THE METHOD: learning with AI, not just copying

Every chapter follows your template: **objectives → simple explanation → technical explanation → real-world example → OpenCV example → onion example → code → expected output → visual → common mistakes → practice → questions+answers**, plus an end-of-chapter block (quick revision, exercise, onion task, AI prompt, debugging challenge, answers).

The AI-assisted coding loop (your "vibe coding" done right):

```text
UNDERSTAND  →  PLAN  →  ASK AI  →  GENERATE  →  RUN  →  TEST  →  DEBUG  →  UNDERSTAND  →  IMPROVE
```

I give you a **ready-to-copy prompt** for a free AI tool at every stage, and I explain the generated code **after** it appears — you are learning, not copy-pasting.

---

# PART F — THE REAL SCREENSHOTS IN THIS BOOK

This book contains **30+ real images and 30+ real pipeline screenshots**, all produced by actually running OpenCV in this environment, including:

- 🖼️ **26 real frames** captured directly from the 13 playlist videos (labeled with the video they came from).
- 📸 **30 real pipeline stages** (original → grayscale → blur → threshold → morphology → contour → mask → segmented → result) for GOOD, DAMAGED and ROTTEN onions.
- 🖼️ **Illustrative** screenshots of VS Code, the terminal and Colab (recreated with the *real* code text so you can read them).

Let's begin.
