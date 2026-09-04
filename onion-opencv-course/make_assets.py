"""Build visual assets for the book:
1. pipeline mosaic (REAL output of running the analyzer)
2. illustrative UI screenshots (VS Code / terminal / Colab) drawn with PIL
3. copy curated REAL playlist frames
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/user/onion-opencv-course"
IMG = os.path.join(ROOT, "images")
ASSETS = os.path.join(ROOT, "book_assets")
os.makedirs(ASSETS, exist_ok=True)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --------------------------------------------------------------------------
# 1) PIPELINE MOSAIC  (real analyzer output)
# --------------------------------------------------------------------------
def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    h = out.shape[0]
    band = np.full((40, out.shape[1], 3), (30, 30, 30), np.uint8)
    cv2.putText(band, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return np.vstack([band, out])

def mosaic(in_dir, out_path):
    names = ["01_original.jpg", "02_grayscale.jpg", "03_blurred.jpg",
             "04_threshold.jpg", "05_morphology.jpg", "06_contour.jpg",
             "07_mask.jpg", "08_masked_onion.jpg", "09_result.jpg"]
    titles = ["1. READ", "2. GRAYSCALE", "3. BLUR", "4. THRESHOLD",
              "5. MORPHOLOGY", "6. CONTOUR", "7. MASK", "8. SEGMENTED",
              "9. RESULT"]
    W = 300
    tiles = []
    for n, t in zip(names, titles):
        p = os.path.join(in_dir, n)
        im = cv2.imread(p)
        h, w = im.shape[:2]
        nh = int(h * W / w)
        im = cv2.resize(im, (W, nh))
        tiles.append(label(im, t))
    rows = []
    for r in range(0, 9, 3):
        row = tiles[r:r+3]
        H = max(t.shape[0] for t in row)
        row = [np.vstack([t, np.full((H - t.shape[0], W, 3), (20, 20, 20), np.uint8)]) for t in row]
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)
    print("mosaic:", out_path, grid.shape)

mosaic(os.path.join(IMG, "pipeline_good"), os.path.join(ASSETS, "pipeline_mosaic_good.jpg"))

# --------------------------------------------------------------------------
# 2) ILLUSTRATIVE UI SCREENSHOTS (PIL)
# --------------------------------------------------------------------------
def code_editor(path):
    W, H = 1160, 620
    img = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(SANS, 14)
    f_small = ImageFont.truetype(SANS, 12)
    f_code = ImageFont.truetype(MONO, 15)
    # title bar
    d.rectangle([0, 0, W, 34], fill=(60, 60, 60))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 24, 10, 24 + i * 24, 20], fill=c)
    d.text((120, 7), "onion_analyzer.py — onion-quality-analyzer — Visual Studio Code",
           fill=(220, 220, 220), font=f_title)
    # sidebar
    d.rectangle([0, 34, 240, H], fill=(37, 37, 38))
    d.rectangle([240, 34, 242, H], fill=(0, 0, 0))
    files = ["onion-quality-analyzer/", "  src/", "    __init__.py", "    analyzer.py", "    features.py", "  images/", "  dataset/", "  main.py", "  requirements.txt", "  README.md"]
    yy = 60
    for fn in files:
        col = (150, 150, 150) if fn.endswith("/") else (212, 212, 212)
        if fn.strip().endswith("analyzer.py"):
            col = (86, 156, 214)
        d.text((18, yy), fn, fill=col, font=f_small)
        yy += 24
    # code area with simple highlighting
    code = [
        ("import", (198, 120, 221)), (" cv2", (220, 220, 220)),
        ("\nimport", (198, 120, 221)), (" numpy ", (220, 220, 220)), ("as", (198, 120, 221)), (" np", (220, 220, 220)),
        ("\n\n", (0,0,0)),
        ("def", (198, 120, 221)), (" segment_onion", (86, 156, 214)), ("(img):", (220, 220, 220)),
        ("\n    gray", (220,220,220)), (" = ", (220,220,220)), ("cv2", (86,156,214)), (".cvtColor(img, ", (220,220,220)),
        ("cv2", (86,156,214)), (".COLOR_BGR2GRAY)", (220,220,220)),
        ("\n    blurred", (220,220,220)), (" = ", (220,220,220)), ("cv2", (86,156,214)),
        (".GaussianBlur(gray, (7, 7), 0)", (220,220,220)),
        ("\n    _, thresh", (220,220,220)), (" = ", (220,220,220)), ("cv2", (86,156,214)),
        (".threshold(blurred, 0, 255,", (220,220,220)),
        ("\n        cv2", (86,156,214)), (".THRESH_BINARY_INV + ", (220,220,220)),
        ("cv2", (86,156,214)), (".THRESH_OTSU)", (220,220,220)),
        ("\n    contours, _", (220,220,220)), (" = ", (220,220,220)), ("cv2", (86,156,214)),
        (".findContours(", (220,220,220)),
        ("\n        thresh, ", (220,220,220)), ("cv2", (86,156,214)), (".RETR_EXTERNAL,", (220,220,220)),
        ("\n        cv2", (86,156,214)), (".CHAIN_APPROX_SIMPLE)", (220,220,220)),
        ("\n    largest", (220,220,220)), (" = ", (220,220,220)), ("max", (86,156,214)),
        ("(contours, key=", (220,220,220)), ("cv2", (86,156,214)), (".contourArea)", (220,220,220)),
        ("\n    return", (198,120,221)), (" gray, blurred, thresh, largest", (220,220,220)),
    ]
    x, y = 262, 56
    for text, col in code:
        for part in text.split("\n"):
            if part == "":
                y += 24
                x = 262
                continue
            d.text((x, y), part, fill=col, font=f_code)
            x += d.textlength(part, font=f_code)
            if x > W - 20:
                y += 24
                x = 262
    # status bar
    d.rectangle([0, H - 22, W, H], fill=(0, 122, 204))
    d.text((10, H - 20), "Ln 34, Col 12    Python 3.13    UTF-8    OpenCV 5.0", fill=(255, 255, 255), font=f_small)
    img.save(path)
    print("ui:", path)

def terminal(path):
    W, H = 1160, 560
    img = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(SANS, 14)
    f_code = ImageFont.truetype(MONO, 16)
    d.rectangle([0, 0, W, 34], fill=(45, 45, 45))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 24, 10, 24 + i * 20, 20], fill=c)
    d.text((110, 7), "user@onion-machine: ~/onion-quality-analyzer", fill=(200, 200, 200), font=f_title)
    lines = [
        ("$ python onion_analyzer.py images/demo_good.jpg images/pipeline_good", (255, 255, 255)),
        ("", (255, 255, 255)),
        ("RESULT: GOOD", (39, 201, 63)),
        ("  area         = 215286.0", (200, 200, 200)),
        ("  perimeter    = 1902.09", (200, 200, 200)),
        ("  circularity  = 0.748", (200, 200, 200)),
        ("  aspect_ratio = 1.217", (200, 200, 200)),
        ("  dark_ratio   = 0.0033", (200, 200, 200)),
        ("  brown_ratio  = 0.0155", (200, 200, 200)),
        ("  defect_ratio = 0.0155  ->  GOOD", (39, 201, 63)),
        ("", (255, 255, 255)),
        ("$ python onion_analyzer.py images/demo_rotten.jpg images/pipeline_rotten", (255, 255, 255)),
        ("RESULT: ROTTEN", (255, 95, 86)),
        ("  dark_ratio   = 0.3219", (200, 200, 200)),
        ("  brown_ratio  = 0.7035", (200, 200, 200)),
        ("  defect_ratio = 0.7035  ->  ROTTEN", (255, 95, 86)),
    ]
    y = 52
    for text, col in lines:
        d.text((18, y), text, fill=col, font=f_code)
        y += 28
    img.save(path)
    print("ui:", path)

def colab(path):
    W, H = 1160, 620
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(SANS, 15)
    f_code = ImageFont.truetype(MONO, 15)
    f_head = ImageFont.truetype(BOLD, 15)
    d.rectangle([0, 0, W, 48], fill=(255, 255, 255))
    d.ellipse([20, 14, 40, 34], fill=(249, 186, 57))
    d.text((52, 12), "Onion_Analyzer.ipynb", fill=(32, 33, 36), font=f_title)
    d.text((W - 240, 12), "File  Edit  View  Insert  Runtime  Tools", fill=(95, 99, 104), font=f_title)
    d.line([0, 48, W, 48], fill=(218, 220, 224))
    # code cell
    d.rectangle([0, 64, W, 300], fill=(247, 247, 247))
    d.text((24, 78), "[1]  # install OpenCV in Colab", fill=(95, 99, 104), font=f_code)
    cell_lines = [
        ("!pip install opencv-python-headless numpy", (188, 76, 160)),
        ("import cv2, numpy as np", (24, 54, 145)),
        ("print(cv2.__version__)", (24, 54, 145)),
    ]
    y = 108
    for text, col in cell_lines:
        d.text((34, y), text, fill=col, font=f_code)
        y += 26
    d.rectangle([0, 300, W, 304], fill=(218, 220, 224))
    # output
    d.text((24, 322), "4.10.0", fill=(60, 64, 67), font=f_code)
    d.rectangle([0, 360, W, 362], fill=(218, 220, 224))
    d.text((24, 380), "[2]  from google.colab.patches import cv2_imshow", fill=(95, 99, 104), font=f_code)
    d.text((34, 408), "cv2_imshow(img)   # show an image inside the notebook", fill=(24, 54, 145), font=f_code)
    d.text((24, 448), "(image appears here)", fill=(154, 160, 166), font=f_code)
    img.save(path)
    print("ui:", path)

code_editor(os.path.join(ASSETS, "ui_vscode.jpg"))
terminal(os.path.join(ASSETS, "ui_terminal.jpg"))
colab(os.path.join(ASSETS, "ui_colab.jpg"))

# --------------------------------------------------------------------------
# 3) CURATED REAL PLAYLIST FRAMES
# --------------------------------------------------------------------------
FRAMES = os.path.join(ROOT, "frames")
picks = {}
for i in range(1, 14):
    p = f"part{i:02d}"
    picks[p] = [f"{p}_00.jpg", f"{p}_06.jpg"]   # title-ish + mid-lecture
for p, fs in picks.items():
    for f in fs:
        src = os.path.join(FRAMES, f)
        if os.path.exists(src):
            os.system(f"cp '{src}' '{ASSETS}/video_{f}'")
print("playlist frames copied:", len(os.listdir(ASSETS)))
