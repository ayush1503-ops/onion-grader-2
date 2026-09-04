"""Build the book: markdown chapters -> single styled HTML -> PDF (WeasyPrint)."""
import os, re, io, base64
import markdown
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from weasyprint import HTML

ROOT = "/home/user/onion-opencv-course"
BOOK = os.path.join(ROOT, "book")
OUT_HTML = os.path.join(ROOT, "onion_quality_analyzer_book.html")
OUT_PDF = os.path.join(ROOT, "onion_quality_analyzer_book.pdf")
COVER = os.path.join(ROOT, "book_assets", "cover.jpg")

FILES = ["00_front.md", "01_ch01_07.md", "02_ch08_13.md", "03_ch14_17.md",
         "04_ch18_20.md", "05_ch21_23.md", "06_ch24_26.md", "07_appendices.md"]

# ---- emoji -> PDF-safe text (DejaVu has no color-emoji glyphs) -------------
EMOJI = {
    "\U0001F9C5": "[ONION]", "\U0001F3A5": "[VIDEO]", "\u2795": "[+]",
    "\U0001F4F8": "[REAL]", "\U0001F5BC": "[ILLUSTRATION]",
    "\u2705": "(ok)", "\u26A0": "(warning)", "\u274C": "(fail)",
    "\U0001F389": "*", "\U0001F3C1": "*", "\u26A1": ">", "\U0001F3CB": ">",
    "\U0001F916": "[AI]", "\U0001F41B": "[BUG]", "\U0001F4BB": "[CODE]",
    "\U0001F4E4": "[OUTPUT]", "\U0001F6E0": "[BUILD]", "\U0001F4DA": "*",
    "\U0001F4A1": "TIP:", "\u2B50": "*", "\U0001F50D": "[CHECK]",
    "\u25B6": ">", "\u2764": ">", "\U0001F4CA": "[DATA]",
    "\U0001F4C8": "[DATA]", "\U0001F9EA": "[TEST]", "\u2728": "*",
    "\uFE0F": "", "\U0001F7E5": "[ ]", "\U0001F7E9": "[x]",
}
def strip_emoji(text):
    for k, v in EMOJI.items():
        text = text.replace(k, v)
    return text

# ---- load + inline images as base64 (downscaled) ---------------------------
def img_data_uri(path, max_w=760):
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        print("  MISSING IMAGE:", path, e)
        return None
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ---- cover image ------------------------------------------------------------
def make_cover():
    W, H = 1100, 1556
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    # vertical gradient
    top, bot = (58, 38, 20), (214, 138, 62)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)],
               fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    # onion illustration (simple, clean)
    ox, oy = W // 2, 560
    d.ellipse([ox - 210, oy - 190, ox + 210, oy + 240], fill=(238, 205, 138), outline=(150, 92, 30), width=6)
    for dx in (-90, 0, 90):
        d.arc([ox - 210, oy - 190, ox + 210, oy + 240], start=260, end=280 + (dx * 0), fill=(150, 92, 30), width=3)
    d.line([(ox, oy - 190), (ox, oy - 240)], fill=(150, 92, 30), width=6)
    d.ellipse([ox - 60, oy - 250, ox + 60, oy - 190], fill=(176, 120, 46))
    for i in range(5):
        yy = oy - 250 + i * 12
        d.line([(ox - 40, yy), (ox + 40, yy)], fill=(120, 78, 24), width=3)
    d.line([(ox, oy + 240), (ox, oy + 300)], fill=(120, 78, 24), width=6)
    for dx in (-60, 0, 60):
        d.line([(ox + dx, oy + 300), (ox + dx + 14, oy + 340)], fill=(120, 78, 24), width=4)
    # text
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
    f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    def center(text, y, font, fill):
        w = d.textlength(text, font=font)
        d.text(((W - w) / 2, y), text, font=font, fill=fill)
    center("ONION QUALITY", 90, f_title, (255, 248, 235))
    center("ANALYZER", 205, f_title, (255, 248, 235))
    d.rectangle([W//2 - 220, 330, W//2 + 220, 336], fill=(255, 248, 235))
    center("The Complete OpenCV + Computer Vision Course Book", 380, f_sub, (255, 240, 220))
    center("From a beginner video playlist to a real working project", 450, f_small, (250, 230, 205))
    center("OpenCV  |  Python  |  Free AI tools  |  Your own dataset", 990, f_small, (250, 230, 205))
    center("A practical, screenshot-rich guide with an AI tutor", 1050, f_small, (250, 230, 205))
    img.save(COVER, "JPEG", quality=90)
    print("cover:", COVER)

# ---- build ----------------------------------------------------------------
def main():
    make_cover()
    raw = "\n\n".join(open(os.path.join(BOOK, f), encoding="utf-8").read() for f in FILES)
    raw = strip_emoji(raw)

    md = markdown.Markdown(extensions=["extra", "sane_lists"])
    html = md.convert(raw)

    # inline images
    def repl(m):
        path = m.group(1)
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print("  MISSING:", full)
            return m.group(0)
        uri = img_data_uri(full, max_w=760)
        return f'src="{uri}"' if uri else m.group(0)
    html = re.sub(r'src="([^"]+)"', repl, html)

    # wrap each h1 block in a section for page breaks
    html = re.sub(r"<h1", "\x00<h1", html)
    parts = html.split("\x00")
    body = ""
    for i, p in enumerate(parts):
        if not p.strip():
            continue
        cls = "chapter first" if i == 0 else "chapter"
        body += f'<section class="{cls}">{p}</section>\n'

    css = """
    @page { size: A4; margin: 19mm 16mm 17mm 16mm;
        @bottom-center { content: counter(page); font-family: 'DejaVu Sans'; font-size: 9pt; color: #999; }
        @bottom-right { content: "Onion Quality Analyzer - OpenCV Course Book"; font-family:'DejaVu Sans'; font-size:7.5pt; color:#bbb; } }
    * { box-sizing: border-box; }
    body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.3pt; line-height: 1.5; color: #201a12; }
    .cover { page-break-after: always; }
    .cover img { width: 100%; }
    section.chapter { page-break-before: always; }
    h1 { font-size: 19pt; color: #7a3b12; margin: 0 0 8pt 0; border-bottom: 3px solid #d98e3f; padding-bottom: 6pt; line-height:1.2; }
    h2 { font-size: 13.5pt; color: #a55a17; margin: 14pt 0 4pt 0; page-break-after: avoid; }
    h3 { font-size: 11.5pt; color: #4a3a26; margin: 10pt 0 3pt 0; page-break-after: avoid; }
    p { margin: 5pt 0; }
    code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f4f0e8; padding: 0 2px; border-radius: 2px; }
    pre { background: #f7f4ee; border: 1px solid #e4dccd; border-radius: 4px; padding: 7pt 9pt; page-break-inside: avoid; }
    pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.35; white-space: pre-wrap; }
    table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 8.9pt; page-break-inside: auto; }
    th, td { border: 1px solid #cfc5b3; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
    th { background: #efe5d3; }
    tr { page-break-inside: avoid; }
    img { max-width: 100%; height: auto; }
    blockquote { border-left: 4px solid #d98e3f; margin: 7pt 0; padding: 2pt 9pt; background: #faf6ef; color: #5a4630; }
    blockquote p { margin: 3pt 0; }
    ul, ol { margin: 4pt 0 4pt 0; padding-left: 18pt; }
    li { margin: 2pt 0; }
    hr { border: none; border-top: 1px solid #d8cdbd; margin: 12pt 0; }
    a { color: #a55a17; text-decoration: none; }
    strong { color: #4a2c10; }
    """
    doc = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
           f'<style>{css}</style></head><body>'
           f'<div class="cover"><img src="{img_data_uri(COVER, max_w=1100)}"></div>'
           f'{body}</body></html>')

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    print("HTML bytes:", len(doc))

    htmldoc = HTML(string=doc, base_url=ROOT)
    htmldoc.write_pdf(OUT_PDF)
    print("PDF pages:", len(htmldoc.render().pages))
    print("PDF written:", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")

if __name__ == "__main__":
    main()
