#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wsgi.py - production entry point for cloud deploys (Render / Railway /
PythonAnywhere / any WSGI host).

Why: `python app.py` runs Flask's built-in DEVELOPMENT server, which is
fine for your laptop but cloud platforms expect a WSGI callable.

Run locally exactly like production would:
    pip install waitress
    python wsgi.py              -> serves on 0.0.0.0:8000

On Render/Railway the platform runs:
    waitress-serve --host=0.0.0.0 --port=$PORT wsgi:app
(or gunicorn on Linux:  gunicorn -b 0.0.0.0:$PORT wsgi:app)
"""

import os

from app import app  # noqa: E402  (the Flask object)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    try:
        from waitress import serve          # cross-platform (Windows too)
        print(f"Production server (waitress) on 0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        print("waitress not installed - falling back to Flask dev server.")
        print("(pip install waitress  for the production server)")
        app.run(host="0.0.0.0", port=port, threaded=True)
