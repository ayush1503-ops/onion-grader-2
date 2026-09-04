#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_models.py - converts trained models to mobile/deployment formats.

  PyTorch .pth  ->  ONNX        (runs anywhere: onnxruntime, Windows, web)
  Keras .keras  ->  TFLite      (runs on Android / Raspberry Pi)

Why export? The SIH app should not need PyTorch/TensorFlow installed to
RUN a model. ONNX and TFLite are tiny runtimes for phones and laptops.

Run:  python export_models.py
"""

import os

import numpy as np

os.makedirs("models", exist_ok=True)


def export_onnx():
    """PyTorch -> ONNX."""
    try:
        import torch
        from pytorch_cnn import TinyOnionCNN
    except ImportError:
        print("torch not installed - ONNX export skipped")
        return
    path = "models/pytorch_onion_cnn.pth"
    if not os.path.exists(path):
        print("train the PyTorch model first (python pytorch_cnn.py)")
        return
    ckpt = torch.load(path, weights_only=False)
    model = TinyOnionCNN(); model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, 96, 96)
    out = "models/pytorch_onion_cnn.onnx"
    torch.onnx.export(model, dummy, out,
                      input_names=["image"], output_names=["scores"],
                      dynamic_axes={"image": {0: "batch"},
                                    "scores": {0: "batch"}})
    print(f"saved -> {out}")

    # prove the export actually works before claiming it
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out)
        scores = sess.run(None, {"image": dummy.numpy()})[0]
        print(f"  verified with onnxruntime: output shape {scores.shape}")
    except ImportError:
        print("  (onnxruntime not installed - skipped the runtime check)")


def export_tflite():
    """Keras -> TFLite."""
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow not installed - TFLite export skipped")
        return
    src = "models/tensorflow_onion_cnn.keras"
    if not os.path.exists(src):
        print("train the TF model first (python tensorflow_cnn.py)")
        return
    model = tf.keras.models.load_model(src)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    out = "models/tensorflow_onion_cnn.tflite"
    with open(out, "wb") as fh:
        fh.write(tfl)
    print(f"saved -> {out}  ({len(tfl)//1024} KB)")

    # verify with the TFLite interpreter
    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    print("  verified with the TFLite interpreter: inference ran OK")


if __name__ == "__main__":
    export_onnx()
    print()
    export_tflite()
