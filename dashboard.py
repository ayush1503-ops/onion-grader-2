#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard.py - the "many batches" view (runs on port 8002).

app.py    = grade ONE batch.
dashboard = see EVERY batch you ever processed: counts, grade mix,
            sizes, flags, trends - so a mandi manager can spot problems
            ("batch 14 has 30% rejects") in one glance.

No JS libraries, no CDN - charts are inline SVG, so this works even in
the sandboxed preview and offline on the local network.

Run:   python dashboard.py
Open:  http://localhost:8002
"""

import glob
import html
import json
import os

from flask import Flask, Response

import api  # reuse the same report-loading logic (single source of truth)

app = Flask(__name__)

COLORS = {"A": "#16a34a", "URS": "#f59e0b", "REJECT": "#dc2626",
          "CHECK": "#0052FF"}


def batches():
    return api.load_reports()


# ---------------------------------------------------------------------------
# tiny inline SVG charts (no external library needed)
# ---------------------------------------------------------------------------
def bar_chart(counts):
    """Vertical bars: onions per batch."""
    if not counts:
        return "<p>No batches yet.</p>"
    mx = max(c for _, c in counts) or 1
    bw, gap, h = 34, 14, 120
    parts = [f'<svg viewBox="0 0 {len(counts)*(bw+gap)} {h+26}" width="100%" '
             f'height="{h+26}" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, c) in enumerate(counts):
        bh = int(c / mx * h)
        x = i * (bw + gap)
        parts.append(f'<rect x="{x}" y="{h-bh}" width="{bw}" height="{bh}" '
                     f'rx="6" fill="#0052FF" opacity="0.85"/>')
        parts.append(f'<text x="{x+bw/2}" y="{h+14}" font-size="9" '
                     f'text-anchor="middle" fill="#64748B">{html.escape(label[-10:])}</text>')
        parts.append(f'<text x="{x+bw/2}" y="{h-bh-4}" font-size="10" '
                     f'text-anchor="middle" fill="#0F172A">{c}</text>')
    parts.append("</svg>")
    return "".join(parts)


def stacked_bar(gp):
    """One horizontal 100% stacked bar: grade mix."""
    segs, x = [], 0.0
    for g in ["A", "URS", "REJECT", "CHECK"]:
        pct = float(gp.get(g, 0) or 0)
        if pct <= 0:
            continue
        segs.append(f'<div style="width:{pct}%;background:{COLORS[g]}" '
                    f'title="{g} {pct}%"></div>')
    inner = "".join(segs) or '<div style="width:100%;background:#e2e8f0"></div>'
    return (f'<div style="display:flex;height:10px;border-radius:99px;'
            f'overflow:hidden;border:1px solid #e2e8f0">{inner}</div>')


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OnionGrader - Batch Dashboard</title>
<style>
 body{{background:#FAFAFA;color:#0F172A;font-family:Inter,system-ui,sans-serif;
       margin:0;line-height:1.55}}
 header{{background:linear-gradient(135deg,#0052FF,#4D7CFF);color:#fff;
         padding:18px 22px}}
 header h1{{margin:0;font-size:1.25rem}}
 header small{{font-family:ui-monospace,monospace;opacity:.85;letter-spacing:.1em}}
 main{{max-width:980px;margin:0 auto;padding:18px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
 .card{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;
        padding:14px 18px;flex:1;min-width:150px}}
 .card b{{font-size:1.6rem;display:block}}
 .panel{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;
         padding:16px;margin-bottom:18px}}
 h2{{font-size:.95rem;margin:0 0 10px}}
 table{{width:100%;border-collapse:collapse;font-size:.85rem}}
 th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid #EEF2F7}}
 th{{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#64748B}}
 .flag{{color:#B45309;font-size:.78rem}}
 .legend span{{display:inline-flex;align-items:center;gap:5px;margin-right:12px;
               font-size:.75rem;color:#475569}}
 .dot{{width:9px;height:9px;border-radius:3px;display:inline-block}}
 a{{color:#0052FF}}
</style></head><body>
<header><h1>📊 Batch Dashboard</h1>
<small>SIH26031 · VISIBLE-SURFACE ANALYSIS ONLY · {n} batches</small></header>
<main>
 <div class="cards">
  <div class="card"><b>{n}</b>batches processed</div>
  <div class="card"><b>{onions}</b>onions graded</div>
  <div class="card"><b>{a_pct}%</b>average Grade A</div>
  <div class="card"><b>{rej_pct}%</b>average Reject</div>
 </div>
 <div class="panel"><h2>Onions per batch</h2>{chart}</div>
 <div class="panel"><h2>All batches</h2>
 <div class="legend">
   <span><i class="dot" style="background:{cA}"></i>Grade A</span>
   <span><i class="dot" style="background:{cU}"></i>URS</span>
   <span><i class="dot" style="background:{cR}"></i>Reject</span>
   <span><i class="dot" style="background:{cC}"></i>Check</span>
 </div>
 <table><tr><th>Batch</th><th>When</th><th>Onions</th><th>Grade mix</th>
   <th>Median mm</th><th>Files</th></tr>{rows}</table>
 </div>
 <div class="panel"><h2>Downloads (integration API)</h2>
  <a href="http://localhost:8001/api/v1/summary.csv">⬇ all_batches_summary.csv</a><br>
  per-batch: <code>/api/v1/batches/&lt;id&gt;/csv</code> and
  <code>/api/v1/batches/&lt;id&gt;/pdf</code> on port 8001
 </div>
 <p style="font-size:.72rem;color:#64748B">Honest limits: grades the VISIBLE
 surface only - no internal rot/moisture detection. Sizes depend on the
 scale reference (coin / distance / assumption). Synthetic demo numbers are
 not real-world accuracy.</p>
</main></body></html>"""


@app.route("/")
def home():
    reps = batches()
    n = len(reps)
    onions = sum(r.get("onion_count", 0) for r in reps.values())
    a_pct = rej_pct = 0.0
    if n:
        a_pct = round(sum(float(r.get("grade_percent", {}).get("A", 0) or 0)
                          for r in reps.values()) / n, 1)
        rej_pct = round(sum(float(r.get("grade_percent", {}).get("REJECT", 0) or 0)
                            for r in reps.values()) / n, 1)
    counts = [(bid, r.get("onion_count", 0)) for bid, r in reps.items()]

    rows = []
    for bid, r in reps.items():
        short = html.escape(bid)
        when = html.escape(str(r.get("timestamp", "")))
        rows.append(
            f"<tr><td><b>{short}</b>"
            f"{'<div class=flag>⚠ ' + html.escape('; '.join(r.get('quality_flags', []))) + '</div>' if r.get('quality_flags') else ''}"
            f"</td><td>{when}</td><td>{r.get('onion_count')}</td>"
            f"<td style='min-width:140px'>{stacked_bar(r.get('grade_percent', {}))}</td>"
            f"<td>{(r.get('diameter_stats') or {}).get('median', '-')}</td>"
            f"<td><a href='http://localhost:8001/api/v1/batches/{short}/csv'>csv</a>"
            f" · <a href='http://localhost:8001/api/v1/batches/{short}/pdf'>pdf</a></td></tr>")

    return Response(PAGE.format(
        n=n, onions=onions, a_pct=a_pct, rej_pct=rej_pct,
        chart=bar_chart(counts), rows="".join(rows),
        cA=COLORS["A"], cU=COLORS["URS"], cR=COLORS["REJECT"], cC=COLORS["CHECK"]),
        mimetype="text/html")


if __name__ == "__main__":
    print("Batch dashboard on http://localhost:8002")
    app.run(host="0.0.0.0", port=8002, debug=False)
