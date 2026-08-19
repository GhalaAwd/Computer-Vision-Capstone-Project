"""
body_language_metrics.py

Member 3's deliverable: Body-Language Metrics (Posture, Gestures, Movement).

Consumes M1's keypoints CSV (see src/pose_utils.py for the contract:
columns frame, person_id, kp_idx, kp_x, kp_y, conf, clip) and produces a
per-second time-series CSV covering:

  1. Posture stability   -- torso angle vs. vertical, sway, stability score
  2. Gesture frequency   -- wrist velocity peak detection, gestures/min
  3. Movement dynamics   -- normalized keypoint displacement / "energy"

Does NOT touch pose inference (M1), tracking/heatmaps (M2), training/eval
(M4), or the dashboard/export (M5). Those are out of scope for M3.

COCO-17 keypoint indices used here (from pose_utils.py docstring):
    5 left_shoulder   6 right_shoulder
    9 left_wrist      10 right_wrist
    11 left_hip       12 right_hip
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --------------------------------------------------------------------------
# Keypoint index constants (COCO-17, matches pose_utils.py docstring)
# --------------------------------------------------------------------------
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
ALL_KP_INDICES = list(range(17))

# --------------------------------------------------------------------------
# Tunable constants -- every one is documented, none are "magic numbers"
# used without explanation.
# --------------------------------------------------------------------------

# Minimum YOLO-pose keypoint confidence to trust a coordinate at all.
# Chosen (not the Ultralytics-default 0.5) because hip keypoints in this
# dataset are frequently cropped/occluded by conference framing -- at 0.5,
# ~70-80% of frames would lose valid hip data entirely (measured on the
# real CSV). 0.3 is a conservative floor that still excludes near-zero,
# effectively-random detections while preserving enough signal to analyze.
# This is a single tunable knob -- callers can override it.
DEFAULT_CONF_THRESHOLD = 0.3

# Stability score decay scale (degrees). stability = 100 * exp(-sway/SCALE).
# SCALE=10 means a 10-degree standard deviation in torso angle over the
# aggregation window maps to a score of 100/e (~36.8), i.e. "10 degrees of
# sway" is treated as the characteristic scale of "notably unstable."
STABILITY_SWAY_SCALE_DEG = 10.0

# Minimum number of valid, matched keypoints required in a frame pair
# before we trust a movement-energy estimate for that frame. Below this,
# a handful of noisy detections could dominate the average.
MIN_VALID_KEYPOINTS_FOR_ENERGY = 3

# Outlier guard for movement energy. M1's presenter-selection rule picks
# the single LARGEST bounding box per frame independently each frame (no
# temporal tracking/ID persistence -- that's M2's job with bytetrack).
# When a nearby audience member is briefly the largest box, "the
# presenter" instantaneously swaps identity for one frame, which looks
# like a huge, physically-impossible keypoint jump -- not real movement.
# We detect this per-clip: any individual keypoint displacement beyond
# median + MOVEMENT_OUTLIER_MAD_MULTIPLIER * MAD (computed over all valid
# per-keypoint displacements in that clip) is treated as a likely
# identity-swap / tracking glitch, and the WHOLE frame's energy is
# excluded (set NaN) rather than silently averaged in. The multiplier is
# deliberately generous (8x) so genuinely fast real gestures/steps are
# not discarded -- only jumps far outside the clip's own movement
# distribution.
MOVEMENT_OUTLIER_MAD_MULTIPLIER = 8.0

# Gesture-detection smoothing window, expressed in seconds of video, not a
# fixed frame count -- this way the *time* covered by the smoothing window
# stays constant regardless of a clip's fps.
GESTURE_SMOOTH_WINDOW_SEC = 0.15

# Minimum time between two counted gesture "events" for the same wrist.
# This is the refractory period that stops one continuous swing of the
# hand from being sliced into many separate gesture counts.
GESTURE_MIN_EVENT_GAP_SEC = 0.3

# Robust threshold multiplier for peak detection: a frame's smoothed
# velocity must exceed median + K * MAD (median absolute deviation) of
# that wrist's own velocity distribution in that clip to count as a
# gesture peak. MAD-based (not mean/std-based) because velocity spikes
# are themselves outliers that would otherwise inflate a std-based
# threshold and make detection less sensitive.
GESTURE_MAD_MULTIPLIER = 2.0

# Interpolate across gaps of at most this many frames when a wrist drops
# below the confidence threshold briefly (e.g. motion blur on a fast
# gesture). Longer gaps are left as real breaks -- we do not invent
# wrist positions across an occlusion of unknown duration.
MAX_INTERP_GAP_FRAMES = 5


# --------------------------------------------------------------------------
# FPS handling -- see README section "FPS handling" for full explanation.
# We never silently assume 30fps. Every clip's fps must be explicitly
# supplied and its provenance is recorded in the output.
# --------------------------------------------------------------------------

@dataclass
class ClipFPS:
    fps: float
    verified: bool
    source_note: str


# clip1/clip2: traced back to the actual Pexels source files linked in
# data/README.md (the mp4 filename Pexels serves reveals the frame rate,
# e.g. ".../8244322/uhd_25fps.mp4").
#
# clip3: originally unverified (source video unavailable at the time).
# Now verified directly by probing the real source file
# '8716582-uhd_3840_2160_25fps.mp4' (Pexels id 8716582) with
# cv2.VideoCapture(...).get(cv2.CAP_PROP_FPS) -> confirmed 25.0 fps,
# 497 frames, 3840x2160, ~19.88s duration. Note: the keypoints CSV for
# clip3 contains 331 frames, not 497 -- M1 evidently processed/kept a
# subset of the source video. That does not affect the fps (frame rate
# is a property of the video, not of how many frames were kept); Member 3
# continues to use the 331 frames actually present in the CSV for all
# calculations, just now with a confirmed-correct fps for converting
# frame numbers to seconds.
FPS_REGISTRY: dict[str, ClipFPS] = {
    "clip1_presenter": ClipFPS(25.0, True, "Verified: Pexels source file 'uhd_25fps.mp4' (clip id 8244322)"),
    "clip2_presenter": ClipFPS(25.0, True, "Verified: Pexels source file '8244304-uhd_2560_1440_25fps.mp4'"),
    "clip3_presenter": ClipFPS(25.0, True, "Verified: probed actual source file "
                                           "'8716582-uhd_3840_2160_25fps.mp4' (Pexels id 8716582) with "
                                           "cv2.VideoCapture(...).get(cv2.CAP_PROP_FPS) == 25.0 "
                                           "(497 frames, 3840x2160, ~19.88s in the source video; "
                                           "the keypoints CSV itself has 331 frames for this clip)."),
}


def get_fps(clip: str, fps_overrides: dict[str, float] | None = None) -> float:
    """
    Resolve the fps to use for a given clip. Never falls back to a silent
    default -- raises if nothing is known and nothing was supplied.

    Priority: explicit fps_overrides > FPS_REGISTRY > cv2 probe of a real
    video file (if video_path is ever wired in) > error.
    """
    if fps_overrides and clip in fps_overrides:
        return float(fps_overrides[clip])
    if clip in FPS_REGISTRY:
        entry = FPS_REGISTRY[clip]
        if not entry.verified:
            warnings.warn(
                f"[body_language_metrics] fps for '{clip}' is UNVERIFIED "
                f"({entry.source_note}). Pass fps_overrides={{'{clip}': <real_fps>}} "
                f"once confirmed."
            )
        return entry.fps
    raise ValueError(
        f"No fps known for clip '{clip}'. Do not assume 30fps silently -- "
        f"supply it via fps_overrides={{'{clip}': <fps>}}, e.g. from "
        f"cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) once the "
        f"source video is available."
    )


def fps_source_note(clip: str, fps_overrides: dict[str, float] | None = None) -> str:
    """Human-readable provenance string for the fps used, for logging/README."""
    if fps_overrides and clip in fps_overrides:
        return f"Explicit override: {fps_overrides[clip]} fps"
    if clip in FPS_REGISTRY:
        return FPS_REGISTRY[clip].source_note
    return "UNKNOWN"


# --------------------------------------------------------------------------
# 1. Loading / validating M1's CSV
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = ["frame", "person_id", "kp_idx", "kp_x", "kp_y", "conf", "clip"]


def load_presenter_keypoints(csv_path: str) -> pd.DataFrame:
    """
    Load and validate Member 1's keypoints CSV. Does not modify or
    re-run pose inference -- purely reads M1's real hand-off.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Keypoints CSV at {csv_path} is missing expected columns "
            f"{missing}. M1's contract requires {REQUIRED_COLUMNS}."
        )
    if (df["person_id"] != "presenter").any():
        # M1's privacy rule guarantees this, but M3 checks rather than trusts.
        bad = df.loc[df["person_id"] != "presenter", "person_id"].unique()
        raise ValueError(
            f"Found non-presenter person_id values {bad!r} in the CSV -- "
            f"this would violate M1's privacy rule (audience must be "
            f"filtered upstream). Refusing to process."
        )
    return df


def _pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape M1's long-format CSV (one row per keypoint) into one row per
    (clip, frame) with kpN_x / kpN_y / kpN_conf columns for every COCO-17
    index. This is an internal helper -- downstream functions consume the
    wide form.
    """
    wide = df.pivot_table(index=["clip", "frame"], columns="kp_idx", values=["kp_x", "kp_y", "conf"])
    field_suffix = {"kp_x": "x", "kp_y": "y", "conf": "conf"}
    wide.columns = [f"kp{idx}_{field_suffix[field]}" for field, idx in wide.columns]
    wide = wide.reset_index().sort_values(["clip", "frame"]).reset_index(drop=True)
    return wide


# --------------------------------------------------------------------------
# 2. Posture stability
# --------------------------------------------------------------------------

def compute_torso_angles(
    wide: pd.DataFrame, conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> pd.DataFrame:
    """
    Per-frame torso angle relative to vertical.

    Formula
    -------
    shoulder_mid = mean(left_shoulder, right_shoulder)
    hip_mid      = mean(left_hip, right_hip)
    torso_vector = shoulder_mid - hip_mid          (points "up the spine")
    angle_deg    = degrees(atan2(dx, dy_up))
        where dx    = torso_vector.x
              dy_up = -(torso_vector.y)   (image y grows downward, so we
                                            flip sign to get "up" as the
                                            zero-angle reference)

    angle_deg == 0   -> perfectly upright torso
    angle_deg > 0    -> leaning toward image-right
    angle_deg < 0    -> leaning toward image-left

    A frame's angle is NaN (not 0, not interpolated) whenever any of the
    four torso keypoints is below `conf_threshold` -- we do not replace
    missing detections with a fabricated "upright" value.
    """
    needed = [
        f"kp{KP_L_SHOULDER}_x", f"kp{KP_L_SHOULDER}_y", f"kp{KP_L_SHOULDER}_conf",
        f"kp{KP_R_SHOULDER}_x", f"kp{KP_R_SHOULDER}_y", f"kp{KP_R_SHOULDER}_conf",
        f"kp{KP_L_HIP}_x", f"kp{KP_L_HIP}_y", f"kp{KP_L_HIP}_conf",
        f"kp{KP_R_HIP}_x", f"kp{KP_R_HIP}_y", f"kp{KP_R_HIP}_conf",
    ]
    missing_cols = [c for c in needed if c not in wide.columns]
    if missing_cols:
        raise ValueError(f"Wide keypoints frame is missing columns needed for posture: {missing_cols}")

    valid = (
        (wide[f"kp{KP_L_SHOULDER}_conf"] >= conf_threshold)
        & (wide[f"kp{KP_R_SHOULDER}_conf"] >= conf_threshold)
        & (wide[f"kp{KP_L_HIP}_conf"] >= conf_threshold)
        & (wide[f"kp{KP_R_HIP}_conf"] >= conf_threshold)
    )

    shoulder_mid_x = (wide[f"kp{KP_L_SHOULDER}_x"] + wide[f"kp{KP_R_SHOULDER}_x"]) / 2.0
    shoulder_mid_y = (wide[f"kp{KP_L_SHOULDER}_y"] + wide[f"kp{KP_R_SHOULDER}_y"]) / 2.0
    hip_mid_x = (wide[f"kp{KP_L_HIP}_x"] + wide[f"kp{KP_R_HIP}_x"]) / 2.0
    hip_mid_y = (wide[f"kp{KP_L_HIP}_y"] + wide[f"kp{KP_R_HIP}_y"]) / 2.0

    dx = shoulder_mid_x - hip_mid_x
    dy_up = -(shoulder_mid_y - hip_mid_y)  # flip: image y grows downward

    angle_deg = np.degrees(np.arctan2(dx, dy_up))
    angle_deg = angle_deg.where(valid, other=np.nan)

    out = wide[["clip", "frame"]].copy()
    out["torso_angle_deg"] = angle_deg
    out["posture_valid"] = valid
    return out


def aggregate_posture_per_window(
    posture_df: pd.DataFrame, second_index: pd.Series
) -> pd.DataFrame:
    """
    Aggregate per-frame torso angles into per-second (second_index) stats:
    mean angle, sway (std-dev of angle within the window), and a bounded
    stability score.

    stability_score = 100 * exp(-sway_std_deg / STABILITY_SWAY_SCALE_DEG)

    This is a monotonically decreasing, bounded-(0,100] function of sway:
    a perfectly still torso (sway=0) scores 100; larger sway asymptotically
    approaches (but never reaches) 0. Windows with fewer than 2 valid
    angle samples cannot define a variance, so sway/stability are NaN
    there (never fabricated as 0 or 100).
    """
    df = posture_df.copy()
    df["second"] = second_index.values

    def _agg(g: pd.DataFrame) -> pd.Series:
        valid_angles = g.loc[g["posture_valid"], "torso_angle_deg"]
        n_valid = len(valid_angles)
        n_total = len(g)
        mean_angle = valid_angles.mean() if n_valid >= 1 else np.nan
        sway_std = valid_angles.std(ddof=0) if n_valid >= 2 else np.nan
        stability = (
            100.0 * np.exp(-sway_std / STABILITY_SWAY_SCALE_DEG)
            if n_valid >= 2 else np.nan
        )
        return pd.Series({
            "posture_frames_total": n_total,
            "posture_frames_valid": n_valid,
            "posture_angle_deg_mean": mean_angle,
            "posture_sway_std_deg": sway_std,
            "posture_stability_score": stability,
        })

    return df.groupby(["clip", "second"]).apply(_agg).reset_index()


# --------------------------------------------------------------------------
# 3. Gesture frequency
# --------------------------------------------------------------------------

def _wrist_series(wide: pd.DataFrame, kp_idx: int, conf_threshold: float) -> pd.DataFrame:
    """Extract one wrist's (frame, x, y) with low-confidence points as NaN."""
    x = wide[f"kp{kp_idx}_x"].copy()
    y = wide[f"kp{kp_idx}_y"].copy()
    conf = wide[f"kp{kp_idx}_conf"]
    invalid = conf < conf_threshold
    x[invalid] = np.nan
    y[invalid] = np.nan
    return pd.DataFrame({"frame": wide["frame"].values, "x": x.values, "y": y.values})


def _velocity_from_positions(pos: pd.DataFrame, fps: float) -> np.ndarray:
    """
    Frame-to-frame speed (pixels/sec) from a position series that may
    contain short NaN gaps (already interpolated up to MAX_INTERP_GAP_FRAMES
    by the caller) and longer real gaps (left as NaN, producing NaN
    velocity there -- we never compute a velocity across an unknown-length
    occlusion).
    """
    dx = pos["x"].diff()
    dy = pos["y"].diff()
    dframe = pos["frame"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    dt = dframe / fps
    velocity = dist / dt
    return velocity.values


def detect_gesture_events(
    wide_clip: pd.DataFrame,
    kp_idx: int,
    fps: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> pd.DataFrame:
    """
    Detect discrete gesture "events" for one wrist within a single clip
    (wide_clip must already be filtered to one clip, sorted by frame).

    Pipeline (each choice documented):
      1. Mask low-confidence positions to NaN (don't trust noisy detections).
      2. Linearly interpolate gaps of <= MAX_INTERP_GAP_FRAMES frames only
         (brief occlusion/motion-blur) -- longer gaps stay NaN and split
         the clip into separate valid segments for peak-finding, so we
         never detect a "peak" that bridges an unknown-length occlusion.
      3. Convert position to velocity (px/s) using this clip's real fps.
      4. Smooth velocity with a rolling mean sized in *seconds*
         (GESTURE_SMOOTH_WINDOW_SEC), not a fixed frame count, so smoothing
         is comparable across clips with different fps.
      5. Within each valid segment, run scipy.signal.find_peaks with:
           - height    = median + GESTURE_MAD_MULTIPLIER * MAD  (adapts to
             that wrist's own noise floor in that clip, robust to outliers)
           - prominence = same threshold (rejects peaks that don't really
             stand out from the local noise floor)
           - distance  = GESTURE_MIN_EVENT_GAP_SEC converted to frames
             (refractory period -- stops one continuous swing of the hand
             from being sliced into multiple "events")

    Returns a DataFrame with one row per detected gesture event:
    columns [frame, time_sec, velocity_px_s].
    """
    pos = _wrist_series(wide_clip, kp_idx, conf_threshold)
    pos["x_interp"] = pos["x"].interpolate(limit=MAX_INTERP_GAP_FRAMES, limit_area="inside")
    pos["y_interp"] = pos["y"].interpolate(limit=MAX_INTERP_GAP_FRAMES, limit_area="inside")
    interp_pos = pd.DataFrame({"frame": pos["frame"], "x": pos["x_interp"], "y": pos["y_interp"]})

    velocity = _velocity_from_positions(interp_pos, fps)

    # Same identity-swap outlier guard as compute_movement_energy: a
    # single-frame "teleport" of the wrist (caused by M1's per-frame
    # largest-bbox presenter rule briefly latching onto someone else) is
    # not a gesture and must not be counted or allowed to bias the
    # adaptive height/prominence threshold below.
    finite_v = velocity[~np.isnan(velocity)]
    if finite_v.size:
        v_median = np.median(finite_v)
        v_mad = np.median(np.abs(finite_v - v_median)) * 1.4826
        v_cutoff = v_median + MOVEMENT_OUTLIER_MAD_MULTIPLIER * v_mad if v_mad > 0 else np.inf
        velocity = np.where(velocity > v_cutoff, np.nan, velocity)

    smooth_window_frames = max(1, round(GESTURE_SMOOTH_WINDOW_SEC * fps))
    velocity_series = pd.Series(velocity)
    smoothed = velocity_series.rolling(window=smooth_window_frames, min_periods=1, center=True).mean()

    frames = interp_pos["frame"].values
    events = []

    # Split into contiguous non-NaN segments so peak detection never
    # spans an occlusion gap.
    is_valid = ~smoothed.isna()
    seg_id = (is_valid != is_valid.shift(fill_value=False)).cumsum()
    for _, seg in pd.DataFrame({"v": smoothed, "seg": seg_id, "valid": is_valid, "frame": frames}).groupby("seg"):
        if not seg["valid"].iloc[0]:
            continue
        v = seg["v"].values
        if len(v) < 3:
            continue
        median = np.median(v)
        mad = np.median(np.abs(v - median)) * 1.4826  # scale MAD to be std-comparable
        if mad == 0:
            continue  # perfectly flat segment, no gesture signal to threshold against
        height = median + GESTURE_MAD_MULTIPLIER * mad
        min_distance_frames = max(1, round(GESTURE_MIN_EVENT_GAP_SEC * fps))
        peak_idx, _ = find_peaks(v, height=height, prominence=height * 0.5, distance=min_distance_frames)
        for pi in peak_idx:
            events.append({
                "frame": seg["frame"].values[pi],
                "time_sec": seg["frame"].values[pi] / fps,
                "velocity_px_s": v[pi],
            })

    return pd.DataFrame(events, columns=["frame", "time_sec", "velocity_px_s"])


# --------------------------------------------------------------------------
# 4. Movement dynamics
# --------------------------------------------------------------------------

def compute_movement_energy(
    wide_clip: pd.DataFrame, fps: float, conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> pd.DataFrame:
    """
    Per-frame "movement energy" for one clip (wide_clip pre-filtered to a
    single clip, sorted by frame).

    Formula
    -------
    For each keypoint i present with conf >= threshold in BOTH frame t and
    frame t-1:
        d_i = euclidean_distance(kp_i[t], kp_i[t-1])
    frame_energy[t] = mean(d_i for valid i) / dt      (px/sec)

    We use the MEAN across valid keypoints, not the sum -- summing would
    make a frame with more visible keypoints look "more energetic" purely
    because more points were detected, which conflates detection recall
    with actual physical movement. Frames with fewer than
    MIN_VALID_KEYPOINTS_FOR_ENERGY valid matched keypoints are set to NaN
    rather than computed from a handful of possibly-noisy points.
    """
    frames = wide_clip["frame"].values
    n = len(frames)
    energy = np.full(n, np.nan)
    valid_kp_count = np.zeros(n, dtype=int)
    frame_max_dist = np.full(n, np.nan)  # for outlier detection, below

    xs = {i: wide_clip[f"kp{i}_x"].values for i in ALL_KP_INDICES}
    ys = {i: wide_clip[f"kp{i}_y"].values for i in ALL_KP_INDICES}
    confs = {i: wide_clip[f"kp{i}_conf"].values for i in ALL_KP_INDICES}

    all_dists_flat = []  # every individual valid keypoint displacement, for the outlier threshold
    per_frame_dists = [None] * n

    for t in range(1, n):
        # only meaningful for truly consecutive frames (dframe should be 1,
        # but compute generally in case of gaps)
        dframe = frames[t] - frames[t - 1]
        if dframe <= 0:
            continue
        dists = []
        for i in ALL_KP_INDICES:
            if confs[i][t] >= conf_threshold and confs[i][t - 1] >= conf_threshold:
                dx = xs[i][t] - xs[i][t - 1]
                dy = ys[i][t] - ys[i][t - 1]
                dists.append(np.sqrt(dx * dx + dy * dy))
        valid_kp_count[t] = len(dists)
        per_frame_dists[t] = dists
        if dists:
            frame_max_dist[t] = max(dists)
            all_dists_flat.extend(dists)

    # Outlier threshold from this clip's own displacement distribution.
    n_outlier_frames = 0
    if all_dists_flat:
        arr = np.array(all_dists_flat)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median)) * 1.4826
        outlier_cutoff = median + MOVEMENT_OUTLIER_MAD_MULTIPLIER * mad if mad > 0 else np.inf
    else:
        outlier_cutoff = np.inf

    for t in range(1, n):
        dframe = frames[t] - frames[t - 1]
        if dframe <= 0 or per_frame_dists[t] is None:
            continue
        dt = dframe / fps
        dists = per_frame_dists[t]
        if len(dists) < MIN_VALID_KEYPOINTS_FOR_ENERGY:
            continue
        if frame_max_dist[t] > outlier_cutoff:
            n_outlier_frames += 1
            continue  # likely presenter-identity swap, not real motion -- excluded
        energy[t] = float(np.mean(dists)) / dt

    out = pd.DataFrame({
        "clip": wide_clip["clip"].values,
        "frame": frames,
        "movement_energy_px_s": energy,
        "movement_valid_keypoints": valid_kp_count,
    })
    out.attrs["n_outlier_frames"] = n_outlier_frames
    return out


# --------------------------------------------------------------------------
# 5. Full per-second pipeline
# --------------------------------------------------------------------------

@dataclass
class ClipReport:
    clip: str
    fps: float
    fps_note: str
    n_frames: int
    duration_sec: float
    posture_valid_frames: int
    posture_invalid_frames: int
    gesture_events_left: int
    gesture_events_right: int
    gestures_per_minute: float
    movement_valid_frames: int
    movement_outlier_frames: int
    posture_stability_mean: float
    movement_energy_mean: float


def generate_metrics(
    csv_path: str,
    fps_overrides: dict[str, float] | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> tuple[pd.DataFrame, list[ClipReport]]:
    """
    Full Member 3 pipeline: load M1's CSV -> per-clip posture / gesture /
    movement metrics -> per-second time-series CSV, ready for M5.

    Returns (metrics_df, reports) where metrics_df is the final per-second
    table and reports is a list of per-clip ClipReport summaries (used by
    the validation step / printed diagnostics).
    """
    long_df = load_presenter_keypoints(csv_path)
    wide = _pivot_wide(long_df)

    all_rows = []
    reports = []

    for clip in wide["clip"].unique():
        wide_clip = wide[wide["clip"] == clip].sort_values("frame").reset_index(drop=True)
        fps = get_fps(clip, fps_overrides)
        fps_note = fps_source_note(clip, fps_overrides)
        n_frames = len(wide_clip)
        duration_sec = n_frames / fps

        # --- Posture ---
        posture_frame_df = compute_torso_angles(wide_clip, conf_threshold)
        second_index = (wide_clip["frame"] // fps).astype(int)
        posture_per_sec = aggregate_posture_per_window(posture_frame_df, second_index)

        # --- Gestures ---
        left_events = detect_gesture_events(wide_clip, KP_L_WRIST, fps, conf_threshold)
        right_events = detect_gesture_events(wide_clip, KP_R_WRIST, fps, conf_threshold)
        total_events = len(left_events) + len(right_events)
        gestures_per_minute = total_events / (duration_sec / 60.0) if duration_sec > 0 else np.nan

        left_events = left_events.assign(wrist="left")
        right_events = right_events.assign(wrist="right")
        all_events = pd.concat([left_events, right_events], ignore_index=True)
        if not all_events.empty:
            all_events["second"] = (all_events["frame"] // fps).astype(int)
            events_per_sec = (
                all_events.groupby(["second", "wrist"]).size().unstack(fill_value=0)
            )
            events_per_sec = events_per_sec.reindex(columns=["left", "right"], fill_value=0)
        else:
            events_per_sec = pd.DataFrame(columns=["left", "right"])

        # --- Movement ---
        movement_frame_df = compute_movement_energy(wide_clip, fps, conf_threshold)
        movement_frame_df["second"] = (movement_frame_df["frame"] // fps).astype(int)
        movement_per_sec = (
            movement_frame_df.groupby("second")
            .agg(
                movement_energy_px_s=("movement_energy_px_s", "mean"),
                movement_frames_valid=("movement_energy_px_s", lambda s: s.notna().sum()),
                movement_frames_total=("movement_energy_px_s", "size"),
            )
            .reset_index()
        )

        # --- Merge posture / gesture / movement per second ---
        all_seconds = pd.DataFrame({"second": np.arange(0, int(np.ceil(duration_sec)))})
        merged = all_seconds.merge(posture_per_sec.drop(columns=["clip"]), on="second", how="left")
        merged = merged.merge(movement_per_sec, on="second", how="left")
        merged["gesture_events_left"] = merged["second"].map(
            events_per_sec["left"] if "left" in events_per_sec else pd.Series(dtype=int)
        ).fillna(0).astype(int)
        merged["gesture_events_right"] = merged["second"].map(
            events_per_sec["right"] if "right" in events_per_sec else pd.Series(dtype=int)
        ).fillna(0).astype(int)
        merged["gesture_events_total"] = merged["gesture_events_left"] + merged["gesture_events_right"]
        merged["gestures_per_minute_clip"] = gestures_per_minute

        merged.insert(0, "clip", clip)
        merged["fps_used"] = fps
        merged["fps_source"] = fps_note

        all_rows.append(merged)

        posture_valid = int(posture_frame_df["posture_valid"].sum())
        reports.append(ClipReport(
            clip=clip,
            fps=fps,
            fps_note=fps_note,
            n_frames=n_frames,
            duration_sec=duration_sec,
            posture_valid_frames=posture_valid,
            posture_invalid_frames=n_frames - posture_valid,
            gesture_events_left=len(left_events),
            gesture_events_right=len(right_events),
            gestures_per_minute=gestures_per_minute,
            movement_valid_frames=int(movement_frame_df["movement_energy_px_s"].notna().sum()),
            movement_outlier_frames=int(movement_frame_df.attrs.get("n_outlier_frames", 0)),
            posture_stability_mean=float(merged["posture_stability_score"].mean(skipna=True)),
            movement_energy_mean=float(merged["movement_energy_px_s"].mean(skipna=True)),
        ))

    metrics_df = pd.concat(all_rows, ignore_index=True)

    column_order = [
        "clip", "second",
        "posture_angle_deg_mean", "posture_sway_std_deg", "posture_stability_score",
        "posture_frames_valid", "posture_frames_total",
        "gesture_events_left", "gesture_events_right", "gesture_events_total",
        "gestures_per_minute_clip",
        "movement_energy_px_s", "movement_frames_valid", "movement_frames_total",
        "fps_used", "fps_source",
    ]
    metrics_df = metrics_df[[c for c in column_order if c in metrics_df.columns]]
    return metrics_df, reports


def print_validation_report(reports: list[ClipReport]) -> None:
    """Print the Step-12 validation summary required by the brief."""
    for r in reports:
        print(f"--- {r.clip} ---")
        print(f"  fps used:              {r.fps}  ({r.fps_note})")
        print(f"  frames:                {r.n_frames}  (duration {r.duration_sec:.2f}s)")
        print(f"  posture valid frames:  {r.posture_valid_frames} / {r.n_frames}"
              f"  ({100*r.posture_valid_frames/r.n_frames:.1f}%)")
        print(f"  posture invalid:       {r.posture_invalid_frames}")
        print(f"  gesture events:        left={r.gesture_events_left}  right={r.gesture_events_right}")
        print(f"  gestures per minute:   {r.gestures_per_minute:.2f}")
        print(f"  movement valid frames: {r.movement_valid_frames} / {r.n_frames}"
              f"  (excluded {r.movement_outlier_frames} as likely presenter-identity-swap outliers)")
        print(f"  mean stability score:  {r.posture_stability_mean:.2f}")
        print(f"  mean movement energy:  {r.movement_energy_mean:.2f} px/s")
        print()
