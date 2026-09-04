"""Mobile-friendly web demo for the onion grader (SIH26031).
Run:  python app.py   ->  open http://<host>:8000 in your phone browser.
"""
import os
from flask import Flask, request, render_template_string
from grader import analyze_batch

app = Flask(__name__)
UPLOAD = "web_uploads"
os.makedirs(UPLOAD, exist_ok=True)

PAGE = """
<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Onion Quality Analyzer</title>
<style>
 body{font-family:system-ui,sans-serif;background:#faf6ef;margin:0;color:#2a1c10}
 .wrap{max-width:640px;margin:auto;padding:16px}
 h1{font-size:22px;color:#7a3b12}
 .card{background:#fff;border:1px solid #e4d8c6;border-radius:10px;padding:14px;margin:12px 0}
 .big{font-size:26px;font-weight:bold;color:#a55a17}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{border-bottom:1px solid #eee;padding:6px 4px;text-align:left}
 .ok{color:#1a7a1a}.bad{color:#c0392b}
 img{width:100%;border-radius:8px;border:1px solid #ddd}
 button{background:#a55a17;color:#fff;border:0;padding:12px 18px;border-radius:8px;font-size:16px}
 input[type=file]{margin:10px 0}
 .note{font-size:12px;color:#8a7a66}
</style></head><body><div class="wrap">
<h1>🧅 Onion Quality Analyzer</h1>
<p class="note">SIH26031 — grades onion quality from one photo (visible surface only).</p>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="img" accept="image/*" required>
  <button type="submit">Analyze batch</button>
</form>
{% if report %}
<div class="card">
  <div class="big">Grade A: {{ report.summary.grade_a }}% &nbsp; URS: {{ report.summary.urs }}% &nbsp; Reject: {{ report.summary.reject }}%</div>
  <p class="note">{{ report.summary.scale }} · {{ report.summary.total }} onions detected</p>
  <table>
    <tr><th>#</th><th>Result</th><th>Diameter</th><th>Grade</th></tr>
    {% for r in report.onions %}
    <tr><td>{{ loop.index }}</td>
        <td class="{{ 'bad' if r.label != 'GOOD' else 'ok' }}">{{ r.label }}</td>
        <td>{{ r.d_mm }} mm</td><td>{{ r.grade }}</td></tr>
    {% endfor %}
  </table>
</div>
<img src="{{ img_url }}">
{% endif %}
</div></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "img" in request.files:
        f = request.files["img"]
        path = os.path.join(UPLOAD, "last.jpg")
        f.save(path)
        ref_mm = float(request.form.get("ref_mm", "27") or "27")
        report, _ = analyze_batch(path, ref_mm, "web_output")
        return render_template_string(
            PAGE, report=report, img_url="/web_output/annotated.jpg")
    return render_template_string(PAGE, report=None, img_url=None)


@app.route("/web_output/<name>")
def out(name):
    from flask import send_from_directory
    return send_from_directory("web_output", name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
