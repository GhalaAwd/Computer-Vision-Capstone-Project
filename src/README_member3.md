# Member 3 -- Body-Language Metrics (Posture, Gestures, Movement)

Value-layer module feeding the M5 dashboard. Consumes Member 1's
`outputs/presenter_keypoints.csv` and produces a per-second time-series
CSV plus reusable plotting helpers.

## Files

- `src/body_language_metrics.py` -- all metric computation + the full pipeline
- `src/body_language_plots.py` -- plotting helpers, consume the metrics CSV only
- `notebooks/03_body_language_metrics.ipynb` -- executed demo notebook
- `outputs/body_language_metrics.csv` -- the deliverable CSV
- `outputs/plot_*.png` -- evidence plots

## Inputs

`outputs/presenter_keypoints.csv` as produced by Member 1's
`extract_presenter_keypoints()` -- columns `frame, person_id, kp_idx,
kp_x, kp_y, conf, clip`, long format (one row per keypoint per frame),
already filtered to the presenter only (M1's privacy rule). This module
never processes audience keypoints and asserts `person_id == "presenter"`
before doing anything else.

## Outputs

`outputs/body_language_metrics.csv`, one row per `(clip, second)`:

| Column | Meaning |
|---|---|
| `clip` | clip identifier from M1's CSV |
| `second` | integer second since the clip started (0-indexed) |
| `posture_angle_deg_mean` | mean torso angle vs. vertical this second |
| `posture_sway_std_deg` | std-dev of torso angle this second (sway) |
| `posture_stability_score` | 0-100, see formula below |
| `posture_frames_valid` / `_total` | how many of this second's frames had a usable torso reading |
| `gesture_events_left` / `_right` / `_total` | discrete gesture events detected this second |
| `gestures_per_minute_clip` | clip-level gesture rate (repeated per row) |
| `movement_energy_px_s` | mean per-second movement energy (px/s) |
| `movement_frames_valid` / `_total` | frames contributing a trusted energy value |
| `fps_used` / `fps_source` | which fps was applied to this clip and why (see below) |

## Formulas

**Posture stability.** Torso vector = shoulder-midpoint minus hip-midpoint
(both midpoints only computed when all 4 keypoints have `conf >= 0.3`).
`angle_deg = degrees(atan2(dx, -dy))`, where `dx`/`dy` are the vector's
x/y components in image coordinates and the y-sign is flipped because
image y grows downward -- so `angle_deg == 0` means a perfectly upright
torso, positive = leaning toward image-right, negative = image-left.

`sway` = the standard deviation (in **degrees**) of the valid per-frame
`angle_deg` values within a given 1-second window. It quantifies how much
the torso angle fluctuated during that second -- a stationary, upright
presenter has `sway ≈ 0°`; a presenter rocking/swaying has a larger
`sway`. `sway` needs >= 2 valid frames in the window to be defined
(variance is undefined from a single point), otherwise it's `NaN`.

`stability_score = 100 * exp(-sway / 10)`

- Units of `sway`: degrees. The score itself is unitless, scaled 0-100.
- Monotonically **decreasing** in `sway` because more angular fluctuation
  should score as less stable -- `exp(-x)` is a strictly decreasing
  function of `x` for `x >= 0`.
- Range: `(0, 100]`. `sway = 0°` -> score `= 100` (perfectly stable).
  As `sway -> ∞`, score `-> 0` but never reaches it (asymptotic, so a
  single very shaky second can't produce a negative or undefined score).
- `10` (`STABILITY_SWAY_SCALE_DEG`) is the *characteristic scale*: a
  `10°` sway maps to `100 * exp(-1) ≈ 36.8`, i.e. "10 degrees of sway in
  a second" is treated as the threshold between "fairly stable" and
  "notably unstable." It's a single named, documented constant --
  adjustable, but not an unexplained magic number.
- Undefined (`NaN`) whenever `sway` itself is `NaN` (fewer than 2 valid
  torso readings that second) -- never defaulted to 0 or 100.

**Gesture frequency.** Wrist velocity (px/s) from frame-to-frame
displacement / dt, smoothed with a rolling mean sized in *time* (0.15s,
fps-independent), then peaks detected with `scipy.signal.find_peaks`
using a per-clip adaptive height/prominence threshold
(`median + 2*MAD` of that wrist's own velocity distribution) and a 0.3s
minimum distance between peaks so one continuous hand swing isn't counted
as multiple gestures.

**Movement dynamics.** A normalized implementation of the original
"total keypoint displacement/energy" requirement:

1. For each of the 17 COCO keypoints, displacement is computed **only**
   between the current and previous frame **for that same keypoint
   index**, and only when both frames have `conf >= 0.3` for it --
   invalid/low-confidence keypoints are excluded from that frame's
   calculation entirely (not zero-filled).
2. All valid per-keypoint displacements in that frame are aggregated by
   taking their **mean** (not sum).
3. The mean is divided by `dt` (elapsed seconds since the previous
   frame) to get a rate in **px/s**.
4. Frames with fewer than `MIN_VALID_KEYPOINTS_FOR_ENERGY = 3` valid
   matched keypoints are left `NaN` -- too little signal to trust.

**Why mean, not sum:** summing displacement across keypoints would make
a frame where YOLO happened to detect more keypoints look "more
energetic" purely because more points were tracked that frame, not
because the presenter moved more. Averaging normalizes for that changing
visibility, so the resulting `movement_energy_px_s` reflects **movement
intensity per keypoint**, i.e. how animated the detected body actually
is, independent of how many joints happened to be confidently detected.
This is a normalization choice within the original displacement/energy
requirement, not a different metric.

## Confidence handling

Every metric uses `conf >= 0.3` as the trust floor (measured against the
real CSV -- a stricter 0.5 would drop 70-80% of hip readings due to
waist-up conference framing). Missing/low-confidence keypoints are left
`NaN`, never replaced with 0 or a fabricated value.

**Known, expected data limitation:** the source presentation clips are
frequently framed waist-up, so hip keypoints (indices 11/12) often fall
below the confidence threshold or are entirely undetected. Posture is
only computed for a second when both shoulders AND both hips clear
`conf >= 0.3` for at least one (mean) / two (sway, stability) frames in
that window. Across the real data this means only **26-45% of frames per
clip** yield a usable torso angle (see per-clip validation numbers
below) -- the rest are correctly `NaN`, not zero, not interpolated, and
not fabricated. This is expected behavior given the framing, not a bug.
Consumers (M5) should treat gaps in the posture columns as "no reliable
reading," not "zero sway" or "perfectly stable."

## Outlier guard (important, real data quality issue found)

M1's presenter-selection rule picks the single largest bounding box
**independently each frame**, with no temporal tracking (that's M2's
`bytetrack` job, out of scope here). When an audience member's box
briefly out-sizes the presenter's, "the presenter" instantaneously swaps
identity for one frame -- this looks like a huge, physically-impossible
keypoint jump, not real movement.

Both `compute_movement_energy()` and `detect_gesture_events()` detect and
exclude these using a robust `median + 8*MAD` outlier cutoff computed
from **that clip's own** displacement/velocity distribution. Properties,
confirmed:

- **Per-clip, not global** -- the median/MAD threshold is recomputed
  independently inside each clip's processing, using only that clip's
  values, so a noisy clip1 cannot affect what counts as an outlier in
  clip2 or clip3 (no cross-clip leakage).
- **Removes spikes only** -- the multiplier (8x) is deliberately generous
  so genuinely fast real gestures/steps are kept; only jumps far outside
  a clip's own normal movement distribution are excluded.
- **Countable** -- the number of excluded frames per clip is tracked
  (`ClipReport.movement_outlier_frames`) and printed by
  `print_validation_report()`.
- **Never touches M1's CSV** -- this is a read-only, in-memory filter
  applied only to Member 3's derived metric arrays; `presenter_keypoints.csv`
  is never modified or overwritten.
- **Not a tracker** -- this guard does not identify, follow, or re-link
  presenter identity across frames; it only decides "is this single
  frame-to-frame jump physically plausible enough to trust for a metric
  value." It is metric-level robustness against pose/presenter-selection
  noise, and is explicitly not a substitute for Member 2's ByteTrack-based
  tracking pipeline.

## FPS handling

No FPS is stored in M1's CSV or notebook, so it had to be established
independently for each clip. Final, fully verified configuration:

| Clip | FPS | Status | Source |
|---|---|---|---|
| `clip1_presenter` | 25.0 | Verified | Pexels source file `uhd_25fps.mp4` (id 8244322) |
| `clip2_presenter` | 25.0 | Verified | Pexels source file `8244304-uhd_2560_1440_25fps.mp4` |
| `clip3_presenter` | 25.0 | Verified | Probed the actual source file `8716582-uhd_3840_2160_25fps.mp4` (Pexels id 8716582) directly with `cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)` -> confirmed `25.0` (source video: 497 frames, 3840x2160, ~19.88s) |

All three clips are now verified — no clip's fps is assumed. `get_fps()`
still never silently defaults to 30fps; it raises if a clip has no known
fps and none is supplied via `fps_overrides`, so this stays safe if a
4th clip is ever added without registering its fps.

**Important distinction:** clip3's *source video* has 497 frames, but
Member 1's keypoints CSV only has 331 frames for `clip3_presenter` (M1
evidently processed/kept a subset). Frame rate is a property of the
video, not of how many frames were kept, so the verified 25 fps applies
regardless -- but Member 3's calculations use the 331 frames actually
present in the CSV, not the source video's 497, for every metric and for
the reported clip duration.

## How Member 5 can consume this

```python
import pandas as pd
metrics = pd.read_csv("outputs/body_language_metrics.csv")
# one row per (clip, second) -- filter by clip, plot/aggregate as needed
```

Or reuse the plotting helpers directly:

```python
from body_language_plots import plot_posture_stability, plot_gesture_frequency, plot_movement_dynamics
plot_posture_stability(metrics)
```

Summary-card numbers (coverage %, gestures/min, stability score) can be
pulled straight from `ClipReport` objects returned by
`generate_metrics()`, or from the `gestures_per_minute_clip` /
`posture_stability_score` columns in the CSV.
