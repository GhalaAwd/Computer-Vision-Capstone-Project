"""
dashboard_data.py

Member 5's integration layer. Loads every upstream deliverable and unifies it
into one in-memory structure that both the static HTML report
(``report_builder.py``) and the Streamlit app (``dashboard.py``) consume.

Nothing here runs a model. It reads the committed artifacts produced by the
other members:

  M1  outputs/presenter_keypoints.csv          (per-keypoint, per-frame; presenter only)
  M2  outputs/{clip}_coverage_stats.json       (spatial coverage, one per clip)
      outputs/{clip}_aggregate_heatmap.png     (movement heatmap image)
      outputs/{clip}_movement_path.png         (path + grid-occupancy image)
  M3  outputs/body_language_metrics.csv         (per-second posture/gesture/movement)
  M4  models/gesture_detector_best.pt           (trained weights; NOT committed - from Drive)
      .../m4_metrics_summary.json               (eval metrics + operating point)

The dashboard is designed to render from the M1/M2/M3 artifacts alone (all
committed), so it works from a clean clone. The M4 model card degrades
gracefully to ``None`` when the summary/weights are not present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# COCO-17 keypoint order (matches src/pose_utils.py). Kept here so the dashboard
# can label keypoints without importing the heavier pose module.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Consistent semantic colours for the three metric families. Used identically in
# the HTML cards, the matplotlib timeline, and the Streamlit app so a colour
# always means the same metric across the whole product.
METRIC_COLORS = {
    "posture": "#2b8a8a",   # teal
    "gesture": "#7b5ea7",   # violet
    "movement": "#d97706",  # amber
    "coverage": "#3e4c59",  # slate
}


# --------------------------------------------------------------------------- #
# Per-clip container
# --------------------------------------------------------------------------- #
@dataclass
class ClipReport:
    """Everything the dashboard needs to render one clip."""
    clip: str
    duration_s: int
    fps: float
    fps_source: str

    # Summary-card figures
    stability_mean: float | None
    gestures_per_minute: float
    gesture_total: int
    movement_energy_mean: float
    coverage_pct: float | None
    distance_px: float | None
    frame_w: int | None
    frame_h: int | None

    # Per-second timeline: columns second, stability, movement_energy, gestures
    timeline: pd.DataFrame

    # Image paths (may be missing on a partial checkout)
    heatmap_png: str | None = None
    movement_path_png: str | None = None

    # Raw upstream blocks, kept for anyone who wants the detail
    coverage_raw: dict = field(default_factory=dict)


@dataclass
class ReportData:
    """The full integrated dataset: every clip plus the shared model card."""
    clips: list[ClipReport]
    model_card: dict | None
    keypoints_available: bool
    outputs_dir: Path
    models_dir: Path


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_metrics(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "body_language_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"M3 metrics not found at {path}. Run notebooks/03_body_language_metrics.ipynb "
            "and commit outputs/body_language_metrics.csv first."
        )
    return pd.read_csv(path)


def _read_coverage(outputs_dir: Path, clip: str) -> dict:
    path = outputs_dir / f"{clip}_coverage_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _find_m4_summary(outputs_dir: Path, models_dir: Path) -> dict | None:
    """Locate M4's metrics summary. It is not committed, so try the likely spots."""
    candidates = [
        outputs_dir / "m4_metrics_summary.json",
        outputs_dir / "m4_custom_training" / "m4_metrics_summary.json",
        models_dir / "m4_metrics_summary.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    return None


def _timeline_for_clip(metrics_df: pd.DataFrame, clip: str) -> pd.DataFrame:
    """Tidy per-second series for one clip. Posture stability keeps its NaN gaps
    (seconds where the torso was not confidently visible) rather than being
    interpolated over - the gaps are honest signal."""
    g = metrics_df[metrics_df["clip"] == clip].sort_values("second")
    return pd.DataFrame({
        "second": g["second"].to_numpy(),
        "stability": g["posture_stability_score"].to_numpy(),
        "movement_energy": g["movement_energy_px_s"].to_numpy(),
        "gestures": g["gesture_events_total"].to_numpy(),
    })


def _summarize_clip(metrics_df: pd.DataFrame, coverage: dict, clip: str,
                    outputs_dir: Path) -> ClipReport:
    g = metrics_df[metrics_df["clip"] == clip]
    stability_valid = g["posture_stability_score"].dropna()

    heatmap = outputs_dir / f"{clip}_aggregate_heatmap.png"
    path_png = outputs_dir / f"{clip}_movement_path.png"

    return ClipReport(
        clip=clip,
        duration_s=int(g["second"].max()) + 1 if len(g) else 0,
        fps=float(g["fps_used"].iloc[0]) if len(g) else 0.0,
        fps_source=str(g["fps_source"].iloc[0]) if len(g) else "",
        stability_mean=float(stability_valid.mean()) if len(stability_valid) else None,
        gestures_per_minute=float(g["gestures_per_minute_clip"].iloc[0]) if len(g) else 0.0,
        gesture_total=int(g["gesture_events_total"].sum()),
        movement_energy_mean=float(g["movement_energy_px_s"].mean()) if len(g) else 0.0,
        coverage_pct=coverage.get("grid_occupancy_pct"),
        distance_px=coverage.get("total_distance_px"),
        frame_w=coverage.get("frame_width"),
        frame_h=coverage.get("frame_height"),
        timeline=_timeline_for_clip(metrics_df, clip),
        heatmap_png=str(heatmap) if heatmap.exists() else None,
        movement_path_png=str(path_png) if path_png.exists() else None,
        coverage_raw=coverage,
    )


def build_report_data(outputs_dir: str | Path = "outputs",
                      models_dir: str | Path = "models") -> ReportData:
    """Assemble the full integrated dataset from the committed artifacts.

    Parameters
    ----------
    outputs_dir, models_dir : str | Path
        Repo-relative locations of the shared outputs/ and models/ folders.

    Returns
    -------
    ReportData
        clips (one ClipReport each), the M4 model card (or None), and flags.
    """
    outputs_dir = Path(outputs_dir)
    models_dir = Path(models_dir)

    metrics_df = _read_metrics(outputs_dir)
    clip_names = list(metrics_df["clip"].drop_duplicates())

    clips = [
        _summarize_clip(metrics_df, _read_coverage(outputs_dir, c), c, outputs_dir)
        for c in clip_names
    ]

    return ReportData(
        clips=clips,
        model_card=_find_m4_summary(outputs_dir, models_dir),
        keypoints_available=(outputs_dir / "presenter_keypoints.csv").exists(),
        outputs_dir=outputs_dir,
        models_dir=models_dir,
    )


if __name__ == "__main__":
    data = build_report_data()
    print(f"Loaded {len(data.clips)} clips | "
          f"keypoints={'yes' if data.keypoints_available else 'no'} | "
          f"model_card={'yes' if data.model_card else 'missing (M4 on Drive)'}")
    for c in data.clips:
        stab = f"{c.stability_mean:.1f}" if c.stability_mean is not None else "n/a"
        cov = f"{c.coverage_pct}%" if c.coverage_pct is not None else "n/a"
        print(f"  {c.clip:18s} {c.duration_s:>3d}s  "
              f"stability={stab:>5s}  gpm={c.gestures_per_minute:>5.1f}  "
              f"move={c.movement_energy_mean:>6.1f}px/s  coverage={cov}")
