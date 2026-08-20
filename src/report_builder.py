"""
report_builder.py

Member 5's post-lecture analytics report. Turns the integrated dataset from
``dashboard_data.build_report_data()`` into a single self-contained HTML file:
summary cards, a synced per-second delivery timeline, and the spatial-coverage
visuals, per clip, plus the M4 model card.

The report is self-contained (all images base64-embedded, fonts from a CDN with
system fallbacks) so it renders inline in Colab and opens anywhere once committed
- no server required. The Streamlit app (``dashboard.py``) is the interactive
companion; this file is the shareable, gradeable artifact.

Design: cool slate neutrals; each metric family keeps one fixed hue everywhere
(posture teal, gesture violet, movement amber); the signature element is the
delivery timeline - one strip per clip showing how the talk unfolded second by
second.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed to render figures
import matplotlib.pyplot as plt
import numpy as np

from dashboard_data import METRIC_COLORS, ReportData, ClipReport, build_report_data


# --------------------------------------------------------------------------- #
# Figure + image helpers
# --------------------------------------------------------------------------- #
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _png_to_base64(path: str | None, max_width: int = 1100) -> str | None:
    """Base64-encode an image, downscaling wide source frames first so the
    self-contained report stays light (the aggregate heatmaps are full 4K frames)."""
    if not path or not Path(path).exists():
        return None
    try:
        from PIL import Image
        img = Image.open(path)
        if img.width > max_width:
            h = round(img.height * max_width / img.width)
            img = img.convert("RGB").resize((max_width, h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            return "JPEG:" + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass  # fall back to embedding the original bytes
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _img_tag_src(b64: str | None) -> str:
    """Turn the encoded string (optionally JPEG-tagged) into a data URI."""
    if b64 is None:
        return ""
    if b64.startswith("JPEG:"):
        return f"data:image/jpeg;base64,{b64[5:]}"
    return f"data:image/png;base64,{b64}"


def delivery_timeline_fig(clip: ClipReport):
    """The signature element: one synced strip showing the talk over time.

    - movement energy  -> amber filled area (right axis)
    - posture stability -> teal line with markers (left axis, 0-100), NaN gaps kept
    - gesture events    -> violet stems along the top, one per second they occur
    """
    t = clip.timeline
    sec = t["second"].to_numpy()
    stability = t["stability"].to_numpy(dtype=float)
    movement = t["movement_energy"].to_numpy(dtype=float)
    gestures = t["gestures"].to_numpy()

    fig, ax_left = plt.subplots(figsize=(11, 2.7))
    ax_right = ax_left.twinx()

    # movement energy - filled area, behind everything
    ax_right.fill_between(sec, movement, color=METRIC_COLORS["movement"],
                          alpha=0.16, zorder=1)
    ax_right.plot(sec, movement, color=METRIC_COLORS["movement"],
                  linewidth=1.4, alpha=0.85, zorder=2, label="Movement energy")
    ax_right.set_ylabel("Movement energy (px/s)", color=METRIC_COLORS["movement"],
                        fontsize=9)
    ax_right.tick_params(axis="y", labelcolor=METRIC_COLORS["movement"], labelsize=8)
    ax_right.set_ylim(bottom=0)

    # posture stability - line only where measured (NaN gaps stay as gaps)
    ax_left.plot(sec, stability, color=METRIC_COLORS["posture"], linewidth=2.2,
                 marker="o", markersize=4, zorder=4, label="Posture stability")
    ax_left.set_ylabel("Posture stability", color=METRIC_COLORS["posture"], fontsize=9)
    ax_left.set_ylim(0, 105)
    ax_left.tick_params(axis="y", labelcolor=METRIC_COLORS["posture"], labelsize=8)
    ax_left.set_xlabel("Seconds into talk", fontsize=9)
    ax_left.set_xlim(sec.min() if len(sec) else 0, sec.max() if len(sec) else 1)

    # gesture events - violet stems along the top edge
    gesture_secs = sec[gestures > 0]
    gesture_counts = gestures[gestures > 0]
    top = 103
    for gs, gc in zip(gesture_secs, gesture_counts):
        ax_left.vlines(gs, top - 6, top, color=METRIC_COLORS["gesture"],
                       linewidth=2.4, zorder=5)
        ax_left.plot(gs, top, marker="v", markersize=7,
                     color=METRIC_COLORS["gesture"], zorder=6)
        if gc > 1:
            ax_left.text(gs, top - 8, f"×{int(gc)}", ha="center", va="top",
                         fontsize=7, color=METRIC_COLORS["gesture"], fontweight="bold")

    # a quiet, honest note when posture had few valid seconds
    if np.isnan(stability).all():
        ax_left.text(0.5, 50, "posture not confidently measurable in this clip",
                     transform=ax_left.transData if False else ax_left.transAxes,
                     ha="center", va="center", fontsize=9, color="#9aa5b1",
                     style="italic")

    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)
    for spine in ("top",):
        ax_left.spines[spine].set_visible(False)
        ax_right.spines[spine].set_visible(False)
    ax_left.spines["left"].set_color(METRIC_COLORS["posture"])
    ax_right.spines["right"].set_color(METRIC_COLORS["movement"])
    ax_left.grid(axis="x", color="#e4e7eb", linewidth=0.6)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --ink:#1f2933; --slate:#3e4c59; --muted:#7b8794; --line:#e4e7eb;
  --surface:#f5f7fa; --card:#ffffff;
  --posture:#2b8a8a; --gesture:#7b5ea7; --movement:#d97706;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:0 24px 72px}
.masthead{padding:52px 0 30px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:'Space Grotesk',sans-serif;text-transform:uppercase;
  letter-spacing:.22em;font-size:12px;font-weight:600;color:var(--muted)}
.masthead h1{font-family:'Space Grotesk',sans-serif;font-weight:600;
  font-size:44px;line-height:1.04;margin:12px 0 8px;letter-spacing:-.01em}
.masthead p{margin:0;color:var(--slate);max-width:60ch}
.legend{display:flex;gap:20px;margin-top:22px;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:7px;font-size:13px;
  color:var(--slate);font-weight:500}
.dot{width:11px;height:11px;border-radius:3px;display:inline-block}
.clip{margin-top:56px}
.clip-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:16px;border-bottom:1px solid var(--line);padding-bottom:12px;flex-wrap:wrap}
.clip-head h2{font-family:'Space Grotesk',sans-serif;font-weight:600;
  font-size:24px;margin:0;letter-spacing:-.01em}
.clip-head .meta{font-size:13px;color:var(--muted);font-family:'Space Grotesk',sans-serif}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 15px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);font-weight:600;margin-bottom:9px}
.card .v{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:600;
  line-height:1;letter-spacing:-.01em}
.card .u{font-size:12px;color:var(--muted);margin-top:5px}
.card.posture{border-top:3px solid var(--posture)}
.card.gesture{border-top:3px solid var(--gesture)}
.card.movement{border-top:3px solid var(--movement)}
.card.coverage{border-top:3px solid var(--slate)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:18px;margin-top:4px}
.panel h3{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;
  margin:0 0 4px;text-transform:uppercase;letter-spacing:.06em}
.panel .cap{font-size:12.5px;color:var(--muted);margin:0 0 14px}
.panel img{width:100%;height:auto;display:block;border-radius:6px}
.spatial{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
.model{margin-top:64px;background:var(--ink);color:#e4e7eb;border-radius:14px;
  padding:32px 30px}
.model h2{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:600;
  margin:0 0 4px;color:#fff}
.model .sub{color:#9aa5b1;font-size:13px;margin:0 0 22px}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.mcard .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:#9aa5b1;font-weight:600;margin-bottom:8px}
.mcard .v{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:600;color:#fff}
.model .op{margin-top:20px;font-size:13px;color:#cbd2d9}
.model .op b{color:#fff;font-family:'Space Grotesk',sans-serif}
.model.absent{background:#f5f7fa;color:var(--slate);border:1px dashed var(--muted)}
.model.absent h2{color:var(--ink)}
.privacy{margin-top:40px;padding:18px 20px;background:#eef4f4;
  border-left:3px solid var(--posture);border-radius:8px;font-size:13.5px;color:var(--slate)}
.foot{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--muted)}
@media(max-width:820px){.cards{grid-template-columns:repeat(2,1fr)}
  .spatial{grid-template-columns:1fr}.mgrid{grid-template-columns:repeat(2,1fr)}
  .masthead h1{font-size:34px}}
"""


def _card(kind: str, label: str, value: str, unit: str = "") -> str:
    u = f'<div class="u">{unit}</div>' if unit else ""
    return (f'<div class="card {kind}"><div class="k">{label}</div>'
            f'<div class="v">{value}</div>{u}</div>')


def _clip_section(clip: ClipReport) -> str:
    stab = f"{clip.stability_mean:.0f}" if clip.stability_mean is not None else "\u2014"
    cov = f"{clip.coverage_pct:.0f}%" if clip.coverage_pct is not None else "\u2014"
    dist = f"{clip.distance_px:,.0f}" if clip.distance_px is not None else "\u2014"
    res = f"{clip.frame_w}\u00d7{clip.frame_h}" if clip.frame_w else "resolution n/a"

    cards = "".join([
        _card("posture", "Posture stability", stab, "/ 100 avg"),
        _card("gesture", "Gestures / min", f"{clip.gestures_per_minute:.0f}",
              f"{clip.gesture_total} total"),
        _card("movement", "Movement energy", f"{clip.movement_energy_mean:.0f}",
              "px/s avg"),
        _card("coverage", "Stage coverage", cov, "of 10\u00d710 grid"),
        _card("coverage", "Distance moved", dist, "px"),
    ])

    timeline_b64 = _fig_to_base64(delivery_timeline_fig(clip))

    heat = _png_to_base64(clip.heatmap_png)
    path = _png_to_base64(clip.movement_path_png)
    spatial_items = []
    if heat:
        spatial_items.append(
            f'<div class="panel"><h3>Movement heatmap</h3>'
            f'<p class="cap">Where the presenter spent time on stage.</p>'
            f'<img src="{_img_tag_src(heat)}" alt="movement heatmap"></div>')
    if path:
        spatial_items.append(
            f'<div class="panel"><h3>Path &amp; grid occupancy</h3>'
            f'<p class="cap">Route walked and cells of the stage visited.</p>'
            f'<img src="{_img_tag_src(path)}" alt="movement path"></div>')
    spatial = f'<div class="spatial">{"".join(spatial_items)}</div>' if spatial_items else ""

    return f"""
    <section class="clip">
      <div class="clip-head">
        <h2>{clip.clip.replace('_', ' ').title()}</h2>
        <span class="meta">{clip.duration_s}s &middot; {clip.fps:.0f} fps &middot; {res}</span>
      </div>
      <div class="cards">{cards}</div>
      <div class="panel">
        <h3>Delivery timeline</h3>
        <p class="cap">Posture stability (teal), movement energy (amber), and gesture
           moments (violet) across the talk. Gaps in the posture line are seconds where
           the torso was not confidently visible.</p>
        <img src="data:image/png;base64,{timeline_b64}" alt="delivery timeline">
      </div>
      {spatial}
    </section>"""


def _model_section(model_card: dict | None) -> str:
    if not model_card:
        return """
    <section class="model absent">
      <h2>Gesture model \u2014 card pending</h2>
      <p class="sub">M4's <code>m4_metrics_summary.json</code> and
      <code>gesture_detector_best.pt</code> live on Drive and are git-ignored. Drop the
      summary into <code>outputs/</code> (or <code>outputs/m4_custom_training/</code>)
      before the final run and this card fills in automatically.</p>
    </section>"""

    m = model_card.get("metrics", {})
    op = model_card.get("chosen_operating_point", {})
    mcards = "".join([
        f'<div class="mcard"><div class="k">mAP@50</div><div class="v">{m.get("mAP50", "\u2014")}</div></div>',
        f'<div class="mcard"><div class="k">mAP@50-95</div><div class="v">{m.get("mAP50_95", "\u2014")}</div></div>',
        f'<div class="mcard"><div class="k">Precision</div><div class="v">{m.get("precision", "\u2014")}</div></div>',
        f'<div class="mcard"><div class="k">Recall</div><div class="v">{m.get("recall", "\u2014")}</div></div>',
    ])
    task = model_card.get("task", "gesture detection").replace("_", " ")
    fam = model_card.get("model_family", "")
    return f"""
    <section class="model">
      <h2>Gesture model \u2014 {task}</h2>
      <p class="sub">{fam} &middot; evaluated on held-out test split</p>
      <div class="mgrid">{mcards}</div>
      <p class="op">Deployed operating point: <b>conf {op.get('conf', '?')}</b>,
         <b>IoU {op.get('iou', '?')}</b> \u2014 chosen recall-favouring, since a missed
         gesture (under-counting engagement) costs the dashboard more than an extra flagged one.</p>
    </section>"""


def build_html_report(data: ReportData, out_path: str | Path,
                      title: str = "MentorVision") -> Path:
    """Render the full integrated report to a single self-contained HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_sections = "".join(_clip_section(c) for c in data.clips)
    model_section = _model_section(data.model_card)
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")

    legend = "".join([
        f'<span><i class="dot" style="background:{METRIC_COLORS["posture"]}"></i>Posture stability</span>',
        f'<span><i class="dot" style="background:{METRIC_COLORS["gesture"]}"></i>Gesture events</span>',
        f'<span><i class="dot" style="background:{METRIC_COLORS["movement"]}"></i>Movement energy</span>',
        f'<span><i class="dot" style="background:{METRIC_COLORS["coverage"]}"></i>Spatial coverage</span>',
    ])

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} \u2014 Delivery Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_CSS}</style></head>
<body><div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Instructor Delivery &amp; Body-Language Report</div>
    <h1>{title}</h1>
    <p>An objective, privacy-preserving read on how a talk was delivered \u2014 posture,
       gestures, movement, and use of the stage \u2014 built from pose estimation over the
       presenter only.</p>
    <div class="legend">{legend}</div>
  </header>
  {clip_sections}
  {model_section}
  <div class="privacy"><b>Privacy by design.</b> Only the presenter is analysed. Every
     clip has an audience visible, but the pipeline keeps just the main speaker
     (largest, most central figure) at the keypoint stage, so no audience member is
     ever measured or stored.</div>
  <div class="foot">Generated {stamp} &middot; MentorVision capstone &middot; built from
     committed pose, tracking, and body-language outputs.</div>
</div></body></html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    data = build_report_data()
    out = build_html_report(data, "outputs/mentorvision_report.html")
    print(f"Report written to {out} ({out.stat().st_size / 1024:.0f} KB)")
