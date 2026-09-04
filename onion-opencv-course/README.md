# 🧅 OnionQuality — AI-based Onion Grading App (SIH26031)

Smart India Hackathon 2026 · Ministry of Consumer Affairs, Food & Public
Distribution · Department of Consumer Affairs (DoCA).

An AI mobile web app that grades a **batch of onions from a single photo**:
detects each onion, classifies **damaged / rotten / sprouted / undersized / good**,
measures size (mm), and instantly reports **Grade A % and URS %** with a digital
quality report — reducing human bias and improving transparency.

## Demo (60 seconds)
1. Open the app (Flask, runs on any phone browser).
2. Place 3–6 onions on a white paper with a coin (₹10 = 27 mm).
3. Take a photo → upload → instant annotated image + grade percentages.

## Run it
```bash
pip install -r requirements.txt
python src/batch_grader.py images/demo_batch.jpg 27   # command line
python app.py                                          # web app -> http://localhost:8000
```

## How it works (pipeline)
`read → resize → grayscale → Gaussian blur → Otsu threshold → morphology →
contours → per-onion mask → features (HSV colour, shape, size, texture) →
rules → GOOD/DAMAGED/ROTTEN/SPROUTED/UNDERSIZED → Grade A / URS / Reject % → report`

- **Size calibration:** a reference coin in the photo (₹10/₹2 = 27 mm, ₹5 = 23 mm,
  ₹1 = 22 mm — configurable in the UI). Without a coin, the app warns and assumes
  a median onion = 55 mm.
- **Grades:** Grade A = 45–65 mm, Grade URS = 35–70 mm (relaxed procurement spec),
  others = Reject. Thresholds are configurable in `src/batch_grader.py`.

## Project layout
```text
src/batch_grader.py    core grading pipeline
app.py                 mobile web UI (upload -> report)
images/demo_batch.jpg  demo photo
batch_output/          annotated image + report (json/txt/card)
SIH26031_pitch_deck.pptx
requirements.txt
```

## Honest limitations
- Grades **visible surface quality only**. A photo cannot detect internal rot,
  internal moisture, or hidden damage.
- Size accuracy depends on the reference object and camera angle (top-down is best).
- Overlapping onions should be spaced apart (watershed segmentation = roadmap item).
- Rule-based now; accuracy improves with a labelled dataset + ML/CNN.

## Requirements
`opencv-python`, `numpy`, `flask`, `pillow`
