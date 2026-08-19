"""
body_language_plots.py

Reusable plotting helpers for Member 3's metrics output. These consume the
per-second metrics DataFrame produced by
`body_language_metrics.generate_metrics()` -- they do not recompute
anything from the raw keypoints CSV. Meant to be reused as-is by Member 5
in the dashboard.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def _clip_axes(metrics_df: pd.DataFrame, clips: list[str] | None = None):
    clips = clips or sorted(metrics_df["clip"].unique())
    fig, axes = plt.subplots(len(clips), 1, figsize=(9, 3 * len(clips)), sharex=False)
    if len(clips) == 1:
        axes = [axes]
    return clips, fig, axes


def plot_posture_stability(metrics_df: pd.DataFrame, clips: list[str] | None = None):
    """Posture stability score (and sway) over time, one subplot per clip."""
    clips, fig, axes = _clip_axes(metrics_df, clips)
    for ax, clip in zip(axes, clips):
        g = metrics_df[metrics_df["clip"] == clip]
        ax.plot(g["second"], g["posture_stability_score"], color="tab:blue", label="Stability score (0-100)")
        ax2 = ax.twinx()
        ax2.plot(g["second"], g["posture_sway_std_deg"], color="tab:orange", alpha=0.6, label="Sway std (deg)")
        ax.set_title(f"Posture stability -- {clip}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stability score", color="tab:blue")
        ax2.set_ylabel("Sway (deg)", color="tab:orange")
        ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def plot_gesture_frequency(metrics_df: pd.DataFrame, clips: list[str] | None = None):
    """Gesture events per second (stacked left/right) over time."""
    clips, fig, axes = _clip_axes(metrics_df, clips)
    for ax, clip in zip(axes, clips):
        g = metrics_df[metrics_df["clip"] == clip]
        ax.bar(g["second"], g["gesture_events_left"], label="Left wrist", color="tab:green")
        ax.bar(g["second"], g["gesture_events_right"], bottom=g["gesture_events_left"],
               label="Right wrist", color="tab:red")
        gpm = g["gestures_per_minute_clip"].iloc[0] if len(g) else float("nan")
        ax.set_title(f"Gesture events -- {clip} ({gpm:.1f} gestures/min)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gesture events")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_movement_dynamics(metrics_df: pd.DataFrame, clips: list[str] | None = None):
    """Movement energy (px/s) over time, one subplot per clip."""
    clips, fig, axes = _clip_axes(metrics_df, clips)
    for ax, clip in zip(axes, clips):
        g = metrics_df[metrics_df["clip"] == clip]
        ax.plot(g["second"], g["movement_energy_px_s"], color="tab:purple")
        ax.fill_between(g["second"], g["movement_energy_px_s"], alpha=0.2, color="tab:purple")
        ax.set_title(f"Movement dynamics -- {clip}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Movement energy (px/s)")
    fig.tight_layout()
    return fig
