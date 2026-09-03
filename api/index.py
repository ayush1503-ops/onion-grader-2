#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api/index.py - VERCEL SERVERLESS ENTRY POINT.

Vercel runs Python from this file (a "serverless function"). It just
flips the app into serverless mode (files embedded in the response,
writable folders in /tmp) and hands over the normal Flask app.

Local use is unchanged:  python app.py
"""

import os

os.environ["VERCEL"] = "1"
os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")
os.environ.setdefault("OUTPUT_DIR", "/tmp/outputs")

from app import app  # noqa: E402  (the Flask app)

# Vercel's free tier allows request bodies up to ~4.5 MB
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
