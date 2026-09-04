#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api.py - integration API for mandi/ERP systems (runs on port 8001).

The MAIN app (app.py, port 8000) is for humans.
THIS api is for OTHER PROGRAMS: it serves every saved batch report as
JSON, CSV (spreadsheets) and PDF (printouts) so a mandi system, a
government portal or a buyer dashboard can pull results automatically.

Endpoints (all GET):
    /api/v1/batches                 -> list of batch summaries (JSON)
    /api/v1/batches/<batch_id>      -> one full report (JSON)
    /api/v1/batches/<batch_id>/csv  -> per-onion table as CSV
    /api/v1/batches/<batch_id>/pdf  -> one-page PDF report
    /api/v1/summary.csv             -> ALL batches in one CSV

Run:   python api.py                       (or: waitress-serve --port=8001 api:app)
Test:  curl http://localhost:8001/api/v1/batches
"""

import csv
import glob
import io
import json
import os

from flask import Flask, Response, abort, jsonify

app = Flask(__name__)
OUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# data access: outputs/ holds files named  <timestamp>_<batch>_report.json
# ---------------------------------------------------------------------------
def _batch_id_from_file(path):
    stem = os.path.basename(path)[:-len("_report.json")]
    # "20260903-105259-893673_test_batch_5_pile" -> "test_batch_5_pile"
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 22 else stem


def load_reports():
    """Newest report per batch_id (old runs of the same batch are ignored)."""
    found = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*_report.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue                          # a broken file never kills the API
        rep["_file"] = os.path.basename(path)
        found[_batch_id_from_file(path)] = rep   # sorted order = newest wins
    return found


def get_report(batch_id):
    rep = load_reports().get(batch_id)
    if not rep:
        abort(404, description=f"batch '{batch_id}' not found")
    return rep


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.route("/api/v1/batches")
def batches():
    out = []
    for bid, r in load_reports().items():
        out.append({"batch_id": bid, "timestamp": r.get("timestamp"),
                    "onion_count": r.get("onion_count"),
                    "grade_percent": r.get("grade_percent", {}),
                    "median_mm": (r.get("diameter_stats") or {}).get("median"),
                    "quality_flags": r.get("quality_flags", [])})
    return jsonify({"count": len(out), "batches": out})


@app.route("/api/v1/batches/<batch_id>")
def batch_detail(batch_id):
    rep = dict(get_report(batch_id))
    rep.pop("_file", None)
    return jsonify(rep)


@app.route("/api/v1/batches/<batch_id>/csv")
def batch_csv(batch_id):
    rep = get_report(batch_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["batch_id", rep.get("batch_id")])
    w.writerow(["timestamp", rep.get("timestamp")])
    w.writerow(["scale_source", rep.get("scale_source")])
    w.writerow([])
    w.writerow(["onion_no", "grade", "defect_class", "diameter_mm",
                "layer", "visibility"])
    for i, o in enumerate(rep.get("onions", []), 1):
        w.writerow([i, o.get("grade"), o.get("defect_class"),
                    o.get("diameter_mm"), o.get("layer"),
                    o.get("visibility")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename={batch_id}.csv"})


@app.route("/api/v1/summary.csv")
def summary_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["batch_id", "timestamp", "onions", "A_%", "URS_%",
                "REJECT_%", "CHECK_%", "median_mm", "weight_kg", "flags"])
    for bid, r in load_reports().items():
        gp = r.get("grade_percent", {})
        w.writerow([bid, r.get("timestamp"), r.get("onion_count"),
                    gp.get("A", 0), gp.get("URS", 0), gp.get("REJECT", 0),
                    gp.get("CHECK", 0), (r.get("diameter_stats") or {}).get("median"),
                    r.get("estimated_weight_kg"),
                    "; ".join(r.get("quality_flags", []))])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=all_batches_summary.csv"})


@app.route("/api/v1/batches/<batch_id>/pdf")
def batch_pdf(batch_id):
    rep = get_report(batch_id)
    buf = io.BytesIO()
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        abort(500, description="reportlab not installed - run: pip install reportlab")

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Onion Batch Report - {batch_id}",
                       styles["Title"]),
             Paragraph(f"Generated {rep.get('timestamp')}  ·  "
                       "visible-surface analysis only (honesty note)",
                       styles["Italic"]),
             Spacer(1, 6 * mm)]

    gp, ds = rep.get("grade_percent", {}), rep.get("diameter_stats") or {}
    head = [["Batch", rep.get("batch_id")],
            ["Onions", rep.get("onion_count")],
            ["Grade A / URS / Rej / Check",
             f"{gp.get('A',0)}% / {gp.get('URS',0)}% / "
             f"{gp.get('REJECT',0)}% / {gp.get('CHECK',0)}%"],
            ["Diameter min-med-max (mm)",
             f"{ds.get('min','-')} - {ds.get('median','-')} - {ds.get('max','-')}"],
            ["Est. weight (kg)", rep.get("estimated_weight_kg")],
            ["Scale basis", rep.get("scale_source")]]
    story += [Table(head, colWidths=[60*mm, 110*mm],
                    style=[("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9"))]),
              Spacer(1, 6 * mm)]

    onions = rep.get("onions", [])
    if onions:
        rows = [["#", "Grade", "Class", "mm", "Layer"]] + [
            [i + 1, o.get("grade"), o.get("defect_class"),
             o.get("diameter_mm"), o.get("layer")]
            for i, o in enumerate(onions)]
        story += [Paragraph("Per-onion results", styles["Heading2"]),
                  Table(rows, colWidths=[15*mm, 30*mm, 40*mm, 30*mm, 30*mm],
                        style=[("GRID", (0, 0), (-1, -1), .3, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E9FF"))])]

    flags = rep.get("quality_flags", [])
    if flags:
        story += [Spacer(1, 4 * mm),
                  Paragraph("Quality flags: " + "; ".join(flags), styles["Italic"])]
    story += [Spacer(1, 4 * mm),
              Paragraph(rep.get("summary", ""), styles["Normal"])]

    doc.build(story)
    pdf = buf.getvalue()
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition":
                             f"attachment; filename={batch_id}.pdf"})


if __name__ == "__main__":
    print("Integration API on http://localhost:8001 "
          "(docs in the file header)")
    app.run(host="0.0.0.0", port=8001, debug=False)
