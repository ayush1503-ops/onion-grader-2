# Chapter 1 — Setting Up Python, OpenCV & Your Free Tool Stack

> **Difficulty:** Beginner · **Prerequisites:** none (you can start here even if you've never coded)
> **Source note:** ➕ This is *setup* knowledge (not a playlist video). The playlist assumes you already have OpenCV installed. We're fixing that gap first.
> **Goal of this chapter:** get Python + OpenCV running on YOUR computer (or in a free browser notebook) and run your first program.

---

## 1. 🎯 Learning Objectives

By the end of this chapter you will be able to:

1. Explain what Python and OpenCV are, in plain words.
2. Choose where to write code: VS Code, Jupyter, or Google Colab (all free).
3. Install Python and OpenCV (or launch a free cloud notebook).
4. Run a "Hello World" image program without errors.
5. Check that OpenCV is installed correctly.

---

## 2. 🧒 Simple Explanation

Think of building software like cooking:

- **Python** is the *kitchen* — the language you use to give instructions ("boil water", "chop onion").
- **OpenCV** is a *box of pre-made cooking tools* — you don't re-invent the knife. OpenCV gives you ready-made functions to read an image, resize it, draw on it, find shapes, etc.
- **An editor/IDE (like VS Code)** is your *workbench* — where you type your instructions.
- **A notebook (Jupyter/Colab)** is a *scratch pad* — you type a bit of code, see the result immediately, repeat.

You don't need to memorize every OpenCV function. You need to know **what's possible** and **how to look it up** (and, later, how to ask AI to explain it).

---

## 3. ⚙️ Technical Explanation

- **Python** is an interpreted, high-level programming language. "Interpreted" means you run the code line-by-line (no messy compile step). "High-level" means it reads almost like English.
- **OpenCV** ("Open Source Computer Vision Library") is a library written mainly in C++ with a Python binding called `cv2`. When you write `import cv2`, you're using those C++ tools from Python. It is free and open-source (BSD license).
- **NumPy** (`import numpy as np`) is the math library OpenCV is built on. **Every OpenCV image is actually a NumPy array.** You'll hear this constantly — remember it.
- **A package manager (`pip`)** downloads and installs libraries for you: `pip install opencv-python numpy`.
- **A virtual environment** is a private, isolated folder for one project's packages, so two projects don't fight each other. Recommended, but we keep it optional for now to stay beginner-friendly.

**The one OpenCV fact to burn into your brain now:** OpenCV loads images in **BGR** order (Blue–Green–Red), not RGB. This confuses everyone at first, and it matters for your onion project later.

---

## 4. 🌍 Real-World Example

OpenCV powers things you already use:

- Your phone camera's face detection (the yellow square around faces).
- Barcode / QR-code scanners.
- Factory machines that inspect products on a conveyor belt — **exactly the same idea as your onion analyzer**: look at an object, decide if it's good or bad.
- License-plate readers, self-driving car vision, Snapchat filters.

Your onion analyzer is the same pattern at small scale: **image in → decision out.**

---

## 5. 🎥 OpenCV Example

The classic "first program" in OpenCV — load an image and show it:

```python
import cv2

img = cv2.imread("onion.jpg")   # read the image
cv2.imshow("My Onion", img)     # show it in a window
cv2.waitKey(0)                  # wait until a key is pressed
cv2.destroyAllWindows()         # close all windows
```

But before you can run this, OpenCV must be installed. That's what the rest of this chapter sets up.

---

## 6. 🧅 Onion Project Example

Your final project will do roughly this:

```text
[ onion photo on disk ]
         ↓  cv2.imread()
[ image loaded into Python ]
         ↓  preprocessing
[ cleaner image ]
         ↓  feature extraction
[ numbers: color, shape, texture ]
         ↓  classification
[ "GOOD"  or  "BAD"  or  "ROTTEN" ]
         ↓  cv2.putText + cv2.imshow
[ image with the result drawn on it ]
```

**Today's job is tiny:** make `cv2.imread()` work and show *any* onion image. The whole pipeline above is built from exactly this first step, repeated and combined.

---

## 7. 💻 Practical Code

### Step 1 — Install Python

- **Windows:** download from [python.org/downloads](https://www.python.org/downloads/). **Important:** tick ✅ "Add Python to PATH" during install, then restart.
- **Mac:** `brew install python` (or download from python.org).
- **Linux:** usually already installed — check with `python3 --version`.

Verify it worked — open a **terminal** and type:

```bash
python --version
# or on Mac/Linux sometimes:
python3 --version
```

Expected: something like `Python 3.12.x`.

### Step 2 — Install OpenCV

In the terminal:

```bash
pip install opencv-python numpy
```

*(If `pip` is not found, try `pip3 install opencv-python numpy`.)*

### Step 3 — Test that OpenCV works

In the terminal, start Python's interactive shell by typing `python`, then:

```python
>>> import cv2
>>> print(cv2.__version__)
4.10.0        # ← your version number may differ
```

If you see a version number with no error → ✅ installed correctly.

### Step 4 — Your first program (save it as a file)

Create a file named `hello_opencv.py`:

```python
# hello_opencv.py  —  your very first OpenCV program
import cv2
import numpy as np

# Make a small blank image (300 x 400, 3 color channels) filled with blue
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (255, 0, 0)          # BGR: lots of Blue, no Green, no Red → blue image

# Put text on it
cv2.putText(img, "Hello Onion!", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

cv2.imshow("First Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Run it from the terminal:

```bash
python hello_opencv.py
```

### Step 5 (alternative) — No install, use Google Colab

If installing is painful, use a free browser notebook instead:

1. Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**.
2. In the first cell, type:

```python
!pip install opencv-python-headless numpy   # the "!" runs a terminal command
import cv2
print("OpenCV version:", cv2.__version__)
```

3. Press **Shift+Enter** to run the cell.

> In Colab you can't pop up windows (`imshow` doesn't work the same way), so you'll use:
> ```python
> from google.colab.patches import cv2_imshow
> cv2_imshow(img)
> ```
> I'll show both ways throughout the book where it matters.

---

## 8. 📤 Expected Output

When you run `hello_opencv.py`:

- A window titled **"First Window"** opens.
- It shows a **blue rectangle** with **yellow text "Hello Onion!"** in the middle.
- The window stays open until you press **any key**.
- After you press a key, the window closes and your terminal returns to the prompt.

That's it — you just loaded the NumPy/OpenCV stack, made an image, drew text, and displayed it.

---

## 9. 🖼️ Visual Explanation

```
Your terminal (where you type commands)
┌──────────────────────────────────────────────────────┐
│ > pip install opencv-python numpy                    │  ← Step 2: install tools
│ > python hello_opencv.py                             │  ← Step 4: run program
│ (program waits here until you press a key...)        │
└──────────────────────────────────────────────────────┘

A window pops up on screen:
┌────────────────────────────────────┐
│  First Window                ─ □ ✕ │
│                                    │
│        ████████████████████        │   ← blue background (255,0,0) in BGR
│        ██  Hello Onion!    ██      │   ← yellow text drawn with putText
│        ████████████████████        │
│                                    │
└────────────────────────────────────┘
        🖼️ Illustrative Screenshot — not a real capture.
```

**What I see → What I do → Why → Expected result**

- **See:** a blue image with yellow text. **Do:** press any key. **Why:** `waitKey(0)` is waiting. **Result:** window closes.
- **See:** `(255, 0, 0)` gave *blue*, not red. **Do:** remember BGR. **Why:** OpenCV stores colors Blue–Green–Red. **Result:** you'll never be surprised by swapped colors again.

---

## 10. ⚠️ Common Mistakes

| Mistake | Why it happens | Fix |
|---------|----------------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed, or wrong Python env | `pip install opencv-python`; make sure you installed into the same Python you're running |
| `python` not recognized on Windows | Python not added to PATH | Re-run installer and tick "Add Python to PATH", restart terminal |
| Window opens and instantly closes | Forgot `cv2.waitKey(0)` | Always put `cv2.waitKey(0)` before `destroyAllWindows()` |
| `error: (-215) ...` / `img is None` | File path wrong or image missing | Print the path; check spelling & folder; see Chapter 3 |
| Red looks blue | Forgot BGR order | Remember: OpenCV = BGR |

---

## 11. 🏋️ Practice Task

1. Install Python + OpenCV (or open Colab).
2. Run `hello_opencv.py`.
3. Change the color `(255, 0, 0)` to `(0, 0, 255)` and predict the new color **before** running. Run it. Were you right?
4. Change the text to your own name and change the font size from `1.0` to `2.0`.

---

## 12. ❓ Questions + Solutions

**Q1.** What is OpenCV?
**A.** A free, open-source library of computer-vision tools, used from Python via `cv2`.

**Q2.** Why do I need NumPy if I'm learning OpenCV?
**A.** OpenCV stores every image as a NumPy array. Understanding NumPy arrays = understanding images.

**Q3.** What does BGR mean and why does it matter?
**A.** OpenCV stores color as Blue–Green–Red instead of the usual Red–Green–Blue. Get the order wrong and colors swap.

**Q4.** What does `cv2.waitKey(0)` do?
**A.** Pauses the program and keeps the image window open until you press a key.

**Q5.** `pip install` fails with "permission denied". What now?
**A.** Try `pip install --user opencv-python`, or use a virtual environment, or use Colab (no install at all).

---

# 📚 End-of-Chapter Block (your Part 18)

## ⚡ Quick Revision

1. What command installs OpenCV?
2. What line imports OpenCV into Python?
3. What order does OpenCV store color channels?
4. What is every OpenCV image technically made of (library)?
5. What function keeps an image window open?
6. What's the difference between a script file and a notebook cell?
7. Name one thing OpenCV is used for in the real world.

## 🏋️ Practical Exercise

Run the following in Colab **and** locally if you can, and write down the version numbers:

```python
import cv2, numpy as np
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
```

## 🧅 Onion Project Task

Take a photo of an onion with your phone. Put it in a folder called `images/` (you'll thank yourself later). Try to load it with `cv2.imread()` and show it. If the path is wrong and you get `None`, fix it and note what the error told you.

## 🤖 AI Practice (copy this into any free AI chat)

> "I'm a complete beginner learning Python and OpenCV. I just installed them. Explain in simple words: (1) what an import statement does, (2) why I type 'cv2.' before function names, and (3) what a NumPy array is. Give me a tiny code example for each. Keep it beginner-friendly and under 200 words."

## 🐛 Debugging Challenge

This program is *supposed* to show a green image, but the window flashes and disappears instantly. Find and fix the two bugs:

```python
import cv2
import numpy as np

img = np.zeros((200, 200, 3), dtype=np.uint8)
img[:] = (0, 255, 0)
cv2.imshow("green", img)
cv2.destroyAllWindows()
```

*(Hint: what's missing between `imshow` and `destroyAllWindows`?)*

## ✅ Answers

**Quick Revision:** 1) `pip install opencv-python` · 2) `import cv2` · 3) BGR · 4) a NumPy array · 5) `cv2.waitKey(0)` · 6) a script runs top-to-bottom all at once; a notebook runs cell-by-cell and shows results inline · 7) face detection, QR scanning, factory product inspection, etc.

**Debugging Challenge fix:** add `cv2.waitKey(0)` before `cv2.destroyAllWindows()`. (The color `(0,255,0)` is actually already correct for green in BGR, so the *only* real bug is the missing `waitKey`.)

---

## 🤖 Your AI tutor for this chapter

Whenever you get stuck, paste the **exact error message** (copy the whole red text) into a free AI chat with this wrapper:

> "I'm a beginner learning OpenCV. Here is the full error I got when running `python hello_opencv.py`:
> [PASTE YOUR ERROR HERE]
> Explain in simple language what caused it, and give me the exact line I should change."

---

**➡️ Next:** [Chapter 2 — What is an image? Pixels, resolution, channels & BGR (Playlist Part 1)](02-chapter-what-is-an-image.md)
