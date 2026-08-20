"""
dashboard.py

MentorVision interactive dashboard. The static HTML report is the shareable
artifact; this is the interactive companion a presenter can click through, clip
by clip.

Run locally:
    streamlit run src/dashboard.py

Run from Colab (needs a tunnel, e.g. localtunnel) - see
notebooks/05_dashboard_export.ipynb.

It reuses the same integration layer (dashboard_data) and the same timeline figure
(report_builder.delivery_timeline_fig) as the HTML report, so a number or a colour
never means two different things across the two surfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable whether Streamlit is launched from repo root or src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_data import build_report_data
from report_builder import delivery_timeline_fig

st.set_page_config(page_title="MentorVision", page_icon="🎤", layout="wide")


@st.cache_data
def load(outputs_dir: str, models_dir: str):
    return build_report_data(outputs_dir, models_dir)


# --- Sidebar: where the outputs live + clip picker ---------------------------
st.sidebar.title("MentorVision")
st.sidebar.caption("Instructor delivery & body-language coach")
outputs_dir = st.sidebar.text_input("outputs/ folder", "outputs")
models_dir = st.sidebar.text_input("models/ folder", "models")

try:
    data = load(outputs_dir, models_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

clip_names = [c.clip for c in data.clips]
selected = st.sidebar.radio("Clip", clip_names,
                            format_func=lambda s: s.replace("_", " ").title())
clip = next(c for c in data.clips if c.clip == selected)

# --- Header ------------------------------------------------------------------
st.title(clip.clip.replace("_", " ").title())
res = f"{clip.frame_w}×{clip.frame_h}" if clip.frame_w else "resolution n/a"
st.caption(f"{clip.duration_s}s · {clip.fps:.0f} fps · {res}")

# --- Summary cards -----------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Posture stability",
          f"{clip.stability_mean:.0f}" if clip.stability_mean is not None else "—",
          help="Mean of the per-second stability score (0-100), over seconds where the torso was visible.")
c2.metric("Gestures / min", f"{clip.gestures_per_minute:.0f}",
          f"{clip.gesture_total} total")
c3.metric("Movement energy", f"{clip.movement_energy_mean:.0f}", "px/s avg")
c4.metric("Stage coverage",
          f"{clip.coverage_pct:.0f}%" if clip.coverage_pct is not None else "—",
          help="Share of a 10x10 stage grid the presenter visited.")
c5.metric("Distance moved",
          f"{clip.distance_px:,.0f}" if clip.distance_px is not None else "—", "px")

# --- Delivery timeline (the signature view) ----------------------------------
st.subheader("Delivery timeline")
st.caption("Posture stability (teal), movement energy (amber), and gesture moments "
           "(violet) across the talk. Gaps in the posture line are seconds where the "
           "torso was not confidently visible.")
st.pyplot(delivery_timeline_fig(clip), use_container_width=True)

# --- Spatial coverage --------------------------------------------------------
st.subheader("Use of the stage")
s1, s2 = st.columns(2)
if clip.heatmap_png:
    s1.image(clip.heatmap_png, caption="Movement heatmap — where time was spent",
             use_container_width=True)
if clip.movement_path_png:
    s2.image(clip.movement_path_png, caption="Path & grid occupancy",
             use_container_width=True)

# --- Model card --------------------------------------------------------------
st.divider()
if data.model_card:
    m = data.model_card.get("metrics", {})
    op = data.model_card.get("chosen_operating_point", {})
    st.subheader(f"Gesture model — {data.model_card.get('task', '').replace('_', ' ')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mAP@50", m.get("mAP50", "—"))
    m2.metric("mAP@50-95", m.get("mAP50_95", "—"))
    m3.metric("Precision", m.get("precision", "—"))
    m4.metric("Recall", m.get("recall", "—"))
    st.caption(f"Deployed operating point: conf {op.get('conf', '?')}, IoU {op.get('iou', '?')} "
               "— recall-favouring, since a missed gesture costs more than an extra flagged one.")
else:
    st.info("Gesture model card pending — add the gesture-model summary "
            "(`gesture_metrics_summary.json`) to the outputs/ folder to populate this section.")

st.caption("Privacy by design: only the presenter is analysed; audience members are "
           "dropped at the keypoint stage.")
