# 🚀 Deployment Guide — OnionGrader (SIH26031)

Five levels, from "demo on my phone" to "app on the Play Store".
Pick the level that matches your need. Times are realistic.

---

## Where each part of this project can live (30-second map)

Everything starts with putting this folder on GitHub — every host
below then deploys *from* GitHub on each push.

| Part | Host | How |
|---|---|---|
| Offline phone app (`offline/`, pure HTML/JS, on-device AI) | **GitHub Pages** — free, HTTPS, no server | bundled `deploy/github-pages.yml` workflow; repo Settings → Pages → Source: *GitHub Actions* (Level 4) |
| Full server app (`app.py`: upload, live camera, reports) | **Render** (free) or **Vercel** | `render.yaml` / `vercel.json` already in repo (Level 2) |
| Integration API (`api.py`: JSON/CSV/PDF) | same Render/Vercel service or its own | Level 2 |
| Quick demo, no hosting at all | tunnel from your laptop | Level 1 |

GitHub Pages can only serve STATIC files — the Python server app must
use Level 2 (Render/Vercel). The two links together = complete product.

---

## Level 0 — Demo on your own Wi-Fi (0 min, already working)

```bash
python app.py
```
Open the printed `http://192.168.x.x:8000` on your phone (same Wi-Fi).
✅ Best for: practicing, small demo with your own devices.

---

## Level 1 — One-command PUBLIC link (5 min) ⭐ best for SIH demos

Show the app to judges on *their* phones without any hosting.

**Option A — Cloudflare Tunnel (free, no account):**
```bash
# install once (Windows: winget install cloudflare.cloudflared / mac: brew install cloudflared)
cloudflared tunnel --url http://localhost:8000
```
It prints a public URL like `https://random-words.trycloudflare.com` —
anyone on Earth can open your app while the command runs.

**Option B — ngrok:**
```bash
ngrok http 8000        # needs a free account at ngrok.com
```

⚠️ Honest notes: the link dies when you stop the command / close the
laptop. The tunnel serves your laptop's files — do the demo from a
laptop with the project on it.

---

## Level 2 — Permanent free cloud URL (15 min) ⭐ recommended

Keeps the app online 24/7 on a free tier. HTTPS included
(**required** for camera access on phones — Level 0 has this problem
solved automatically).

### Vercel (serverless)

This repo ships `app.py` (Flask entrypoint) + `vercel.json` for Vercel.
It uses **uv with `pyproject.toml` + `uv.lock`** (Python 3.12) — the
modern, reliable path. Vercel auto-detects those files and runs
`uv sync --active --no-dev --link-mode hardlink --locked --no-editable`.

⚠️ **Why Python 3.12?** Vercel only supports **3.12 (default), 3.13 and
3.14** — not 3.11, and not arbitrary patch versions. Two separate
pitfalls have bitten this project:

1. **Exact old patch (`3.13.4`)** — uv's managed Python store keeps only
   the *latest* patch per minor (e.g. 3.13.15, 3.12.14). An exact pin
   like `3.13.4` doesn't exist, so uv fails:
   > `error: No interpreter found for Python 3.13.4 in managed installations`

2. **Unsupported minor (`3.11`)** — even a bare `3.11` fails on Vercel,
   because Vercel only installs 3.12/3.13/3.14. uv can't find a 3.11
   interpreter (and can't download one during the build), so the same
   error comes back:
   > `error: No interpreter found for Python 3.11 in managed installations`

Fix: pin a **supported** version. This repo pins `3.12` in
`.python-version`, which satisfies `requires-python = ">=3.11,<3.13"`.
Render stays on 3.11.11 (`render.yaml`) — it installs with `pip`, not uv,
and both 3.11 and 3.12 satisfy the project range.

⚠️ **`vercel.json` `includeFiles` must cover the Python modules.**
`app.py` does `import grader` / `import yolo_mode`, but `includeFiles`
is an *allowlist* — anything not matched is left out of the function
bundle (Vercel does no import tracing / tree-shaking for Python). The
glob must include `**/*.py` (or at least `grader.py` + `yolo_mode.py`),
otherwise the build succeeds but every request 500s with
`ModuleNotFoundError: No module named 'grader'`.

⚠️ **Do NOT add `rewrites` to `vercel.json`** (e.g. the old
`/(.*) → /app.py` catch-all). Vercel now routes internal rewrites in
backend-framework (Flask/Python) projects using the rewritten
destination path, so Flask would receive `/app.py` as the request path
for every request and 404 everything. Vercel's Flask framework build
already sends ALL requests to `app.py` with their original paths —
no rewrites are needed.

If you use `pip` only, `requirements.txt` still works — Render and
local `pip install -r requirements.txt` both work. `pyproject.toml`
lists the same 5 pinned deps (Flask, opencv-headless, numpy, pillow,
waitress) for uv.

### Render.com (free, easiest)
1. Put the project on **GitHub** (create a repo, upload the folder —
   do NOT upload `uploads/`, `outputs/`, `models/`, `runs/` — add a
   `.gitignore`).
2. On render.com → **New → Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `waitress-serve --host=0.0.0.0 --port=$PORT wsgi:app`
   - **Instance type:** Free
4. Deploy → you get `https://your-app.onrender.com` forever.

⚠️ Honest notes about the free tier:
- The free instance **sleeps** after ~15 min idle → the first visitor
  waits ~1 min while it wakes.
- **Don't enable YOLO mode on the free cloud** — the model + torch is
  too heavy for a tiny free instance. The classic CV mode is light and
  works fine. Train/run YOLO locally or on Colab.
- analysis happens on the server: uploads count against bandwidth.

### Alternatives: Railway.app, PythonAnywhere, Fly.io (same idea).

---

## Level 3 — Install on a phone as an APP (10 min, no store)

The `offline/` folder is a **PWA** (Progressive Web App). It runs the
whole AI **on the phone** — no server, works with **no internet**.

### Android / Chrome:
1. Deploy `offline/` to any **HTTPS** static host (Level 4 below), OR
   open it from your laptop once (`http://192.168.x.x:8000/offline/`).
2. Open the URL in Chrome → menu (⋮) → **"Add to Home screen / Install
   app"**.
3. An app icon appears. Open it like any app. Turn OFF Wi-Fi — it still
   works (service worker + on-device AI).

### iPhone / Safari:
Open the URL → Share → **"Add to Home Screen"**. (iOS installs PWAs
the same way; camera works over HTTPS.)

---

## Level 4 — Free hosting for the OFFLINE app (5 min)

The offline app needs **no Python server at all** (it's pure HTML/JS),
so static hosting is free and instant:

**Netlify Drop (easiest):** go to app.netlify.com/drop and drag the
`offline/` folder in → instant HTTPS URL → install from there.

**GitHub Pages (automated — this repo ships the workflow):**
1. The repo ships `deploy/github-pages.yml`, a ready Actions workflow
   that publishes the `offline/` folder as a static site. GitHub does
   not let bots add workflow files, so move it into place once (or
   upload it to `.github/workflows/` via the web UI):
   ```bash
   mkdir -p .github/workflows
   git mv deploy/github-pages.yml .github/workflows/pages.yml
   git commit -m "ci: enable GitHub Pages deploy" && git push
   ```
2. On GitHub: repo **Settings → Pages → Build and deployment →
   Source: "GitHub Actions"** (one-time, 10 seconds).
3. Push to `main` (or Actions → *Deploy offline PWA to GitHub Pages*
   → Run workflow) → done: `https://<your-user>.github.io/<repo>/`
4. Install the PWA from that URL exactly as in Level 3.

No second repo needed any more — the workflow serves `offline/`
straight from this repository, and re-deploys on every change to it.

⚠️ Camera on a phone requires **HTTPS** — both options above give it
automatically. `file://` (double-clicking index.html) will NOT enable
the camera, but photo upload still works.

---

## Level 5 — Real Android APK / Play Store (1–2 hrs)

Turn the installed PWA into a **real APK**:

1. Deploy the offline app to HTTPS first (Level 4).
2. Go to **pwabuilder.com** → paste your URL → **Package for Android**.
3. Download the APK/AAB (Trusted Web Activity wrapper — it's your app,
   full screen, own icon, no browser bar).
4. Install the APK on any Android phone ("allow unknown apps" for
   sideloading), or publish the AAB on the Play Store
   (one-time $25 developer account).

⚠️ Honest notes: a TWA is your web app wrapped as an app — for SIH
this is completely legitimate and standard practice. A fully native
(Kotlin) rewrite is NOT needed and adds months of work.

---

## What to deploy for SIH judges (our recommendation)

| Need | Use |
|---|---|
| Live demo everyone can open | **Level 1** tunnel from your laptop |
| "Here's the real product" link | **Level 2** Render (classic CV mode) |
| "It works offline in villages" | **Level 3/4** offline PWA on the phone |
| "It's a real app with an icon" | **Level 5** APK via PWABuilder |

Demo flow: judges scan the QR of the tunnel URL → open Live Scan →
point at onions → boxes appear → Generate report → show the one-file
JPEG report. Then show airplane mode + the offline app. That wins.

---

## Before you push to GitHub — .gitignore

Create `.gitignore` in the project root:

```
uploads/
outputs/
runs/
models/*.pt
snapshot_*.jpg
__pycache__/
*.pyc
test_images/
```

(model files and user photos don't belong in git — keeps the repo small)

---

## Honest limitations (keep saying these)

- Grades **visible surface only** — no internal rot/damage/moisture
  detection is possible from a normal photo. Anywhere. Ever.
- Size accuracy depends on the reference: coin = exact mm, no coin =
  ±20% (distance mode) or rough estimates.
- No accuracy numbers are claimed anywhere until YOU measure them on a
  labeled test set — this remains true in production.
