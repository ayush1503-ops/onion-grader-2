# 🎓 CNN Training Guide — Laptop CPU & Google Colab (free T4 GPU)

This project trains **three** image classifiers on onions:

| File | What it is | Why it matters |
|---|---|---|
| `pytorch_cnn.py` | small custom CNN, **PyTorch** | the from-scratch baseline |
| `tensorflow_cnn.py` | same idea, **TensorFlow/Keras** | proves the pipeline in framework #2 |
| `transfer_learning.py` | **MobileNetV2 pretrained** (Keras) | the one you'll actually ship |

**The measured lesson (synthetic demo data, 800 images, laptop CPU):**

| Model | Test accuracy (SYNTHETIC demo only!) |
|---|---|
| PyTorch custom CNN | 0.767 |
| TensorFlow custom CNN | 0.917 |
| Transfer learning (MobileNetV2) | **1.000** |

⚠️ These are **synthetic-data numbers**. They prove the pipeline works and that
transfer learning learns faster — they are **NOT real-farm accuracy**.
Never quote them as real-world performance.

---

## 1 · On your laptop (CPU, ~5 minutes total)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu onnx onnxruntime onnxscript

python make_dataset.py          # draws 800 synthetic labeled onions
python pytorch_cnn.py           # trains + saves models/pytorch_onion_cnn.pth
python tensorflow_cnn.py        # trains + saves models/tensorflow_onion_cnn.keras
python transfer_learning.py     # trains + saves models/transfer_mobilenetv2.keras
python evaluate.py              # confusion matrices on the TEST set
python export_models.py         # -> .onnx + .tflite (verified)
```

## 2 · On Google Colab (free T4 GPU, for BIGGER data)

1. Go to colab.research.google.com → New notebook.
2. **Runtime → Change runtime type → T4 GPU.**
3. Upload your real photos as a zip (or mount Drive):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Arrange photos exactly like our layout (folder name = label):
   ```
   dataset/train/good/*.jpg   dataset/train/sprouted/*.jpg  ...
   dataset/val/<class>/*.jpg  dataset/test/<class>/*.jpg
   ```
   (Tip: our `make_dataset.py` shows the 70/15/15 split logic — keep it.)
5. Install & train — same files, GPU makes it 10–50× faster:
   ```python
   !pip -q install tensorflow onnx onnxruntime onnxscript
   !python transfer_learning.py
   !python evaluate.py
   ```
6. Download `models/transfer_mobilenetv2.keras` + the `.tflite` export.

**Colab notes (honest):**
- Free T4 sessions **time out** (~a few hours, idle disconnects sooner) —
  save to Drive often.
- 96×96 images train fine for demo; real photos usually want 160–224 px
  and more epochs (10–30) — still minutes on a T4.
- Same-file rule: keep `dataset/<split>/<class>/` layout and nothing else
  needs to change.

## 3 · Swapping in REAL photos (the day you have them)

1. Collect photos per class (aim 200+ per class to start; more is better).
2. Split 70/15/15 into train/val/test folders — **never** let the same
   farm/batch appear in two splits (that fakes accuracy).
3. Re-run the training scripts unchanged.
4. `evaluate.py` prints the new confusion matrix — quote THOSE numbers,
   and quote them as measured-on-your-test-set.

## 4 · Reading a confusion matrix (30-second version)

Rows = truth, columns = the model's guess. The diagonal = correct.
Our PyTorch run guessed `good → cut` 18 times: the pale cut line on a
small onion is genuinely subtle at 96 px — exactly the kind of insight
confusion matrices exist to reveal. Fix with more/better data or bigger
input size, not by tuning the test set (never touch test for training).
