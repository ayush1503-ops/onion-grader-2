# 🚀 Deployment Guide — OnionGrader (SIH26031)

Five levels, from "demo on my phone" to "app on the Play Store".
Pick the level that matches your need. Times are realistic.

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

This repo ships `api/index.py` + `vercel.json` for Vercel. It installs
with **plain pip from `requirements.txt`**.

⚠️ **Do NOT add `uv.lock` / `pyproject.toml` back.** Vercel's Python
runtime auto-detects those files and force-runs its own
`uv sync --active --no-dev --link-mode hardlink --frozen --no-editable`
(which ignores our `installCommand` and fails in Vercel's build). They
were deliberately removed so Vercel uses `pip install -r requirements.txt`.
If you use `uv` locally, keep those two files git-ignored — never commit
them. All runtime dependencies live in `requirements.txt`.

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

**GitHub Pages:**
1. Create repo, copy the contents of `offline/` into it
   (index.html, opencv.js, sw.js, manifest, icons — all at repo root).
2. Settings → Pages → deploy from branch → done:
   `https://yourname.github.io/repo/`

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
