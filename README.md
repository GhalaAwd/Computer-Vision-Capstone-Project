# MentorVision

**MentorVision** is an **instructor delivery and body-language coaching system** that turns a lecture recording into a privacy-preserving, post-lecture analytics report on how the talk was delivered. It is built with Ultralytics YOLO11 pose estimation, ByteTrack multi-object tracking, OpenCV video analytics, a custom-trained YOLO11 gesture detector, ONNX export, and a Streamlit dashboard.

> **Students:** Aldanah Althenyan, Dalia Alosaimi, Ghala Alawad, Jory Alhassan, Shahad Alahmari
>
> **Training program:** Computer Vision for Developers
>
> **Delivered by:** SDAIA Academy
>
> **Trainer:** Mohammed Albeladi
>
> **Cohort/session dates:** **16 August - 20 August 2026**
>
> **SDAIA Academy on GitHub:** https://github.com/SDAIAAcademy

---

## Project Description

**MentorVision** performs the assessment a speaking coach would give after a lecture: watching how the presenter stood, moved, gestured, and used the room, and turning that into concrete, objective feedback instead of a general impression.

Lecturers and speakers rarely get data-driven feedback on their physical delivery. A colleague sitting in the back can say a talk felt static or restless, but cannot say the presenter held one spot for the first eight seconds, covered nine percent of the stage, or gestured eleven times a minute. Delivery is judged by memory and gut feeling, which is subjective, hard to compare across talks, and easy to disagree about. Video alone does not solve this either, because rewatching a recording is slow and still leaves the reviewer estimating rather than measuring.

MentorVision addresses this by treating a lecture clip as a measurable signal and dividing the analysis into focused computer-vision stages, each producing a documented artifact that the next stage builds on. The system does not rely on one model to do everything. Instead, each stage has a single responsibility, a task-specific model, and an explicit output that the following stages consume.

The pipeline begins with **pose estimation**. A YOLO11 pose model runs real inference over each clip and extracts the presenter's seventeen body keypoints per frame. Because every clip has an audience visible, a privacy rule keeps only the main presenter (the largest, most central figure) and drops everyone else at this stage, so no audience member is ever measured or stored. The result is a per-frame keypoints table that is the shared contract for the rest of the system.

Two stages then read that table in parallel. A **tracking, heatmap, and spatial-coverage** stage follows the presenter with ByteTrack inside a real OpenCV capture-process-write pipeline, renders a movement heatmap with the Ultralytics solutions module, and measures how much of the room the presenter used through a grid-occupancy metric, a total distance travelled, and a movement path. A **body-language metrics** stage converts the same keypoints into three per-second time series: posture stability from the torso angle, gesture frequency from wrist motion, and movement energy from overall keypoint displacement.

Running alongside these is an independent **custom training and evaluation** stage. A YOLO11 detector is fine-tuned on a hand-gesture dataset across two documented runs, a baseline and a tuned run with a frozen backbone and softened augmentation, then evaluated on a held-out test split with mean average precision, precision, recall, and a confusion matrix, and an operating point justified through a confidence and IoU threshold sweep.

The final **integration, report, and export** stage unifies every upstream artifact into one dataset, exports the trained gesture model to ONNX for lightweight deployment, and renders the post-lecture analytics dashboard: summary cards, a synced delivery timeline that shows posture, movement, and gesture moments second by second, and the spatial-coverage visuals, delivered both as a self-contained HTML report and as an interactive Streamlit app.

For evidence, the repository contains executed Colab notebooks with captured output for every stage, the committed analysis artifacts, and the shared Python modules that the notebooks import.

---

## Problem Statement

Assessing how a lecture was delivered is normally done by memory and impression, which is subjective and impossible to compare fairly across talks.

Handled that way, or by simply rewatching the recording, this commonly leads to:

- Delivery judged by feel, with no numbers behind "too static" or "too restless"
- Feedback that cannot be compared between two talks or tracked over time
- Movement, posture, and gestures estimated rather than measured
- Use of the room described vaguely instead of quantified
- Audience members captured in any automated analysis, raising a privacy concern
- Findings that live only in a reviewer's head, with no report to revisit

MentorVision provides one pipeline that measures physical delivery from the recording, keeps the analysis to the presenter alone, and produces a shareable report that a speaker can read and act on.

---

## System Objectives

The system is designed to:

1. Run real pose inference over a lecture clip and extract the presenter's body keypoints per frame.
2. Keep only the presenter and drop the audience at the keypoint stage, in code, for privacy.
3. Track the presenter with a stable identity through a real OpenCV video pipeline.
4. Render a movement heatmap and measure spatial coverage, distance, and movement path.
5. Measure posture stability, gesture frequency, and movement energy as per-second time series.
6. Fine-tune a gesture detector on a documented dataset across two deliberate training runs.
7. Evaluate the detector on a held-out test split and report mean average precision, precision, recall, and a confusion matrix.
8. Justify the deployed confidence and IoU thresholds with a measured sweep rather than intuition.
9. Unify every stage's output into a single integrated dataset.
10. Export the trained model to an optimized format for lightweight deployment.
11. Produce a post-lecture analytics report and an interactive dashboard from the integrated data.

---

## Pipeline Architecture

The system uses a **staged pipeline with file-based hand-offs between stages**.

Each stage is a self-contained notebook with one responsibility and a task-specific model. Stages do not call each other directly. Instead, every stage writes a documented artifact to the shared `outputs/` folder, and the next stage reads it. This keeps the stages independent, individually runnable, and individually verifiable, and it is what lets the dashboard render from committed data without re-running any model.

### Data Contract

The presenter keypoints table is the contract the rest of the system is built on. Its columns (`frame`, `person_id`, `kp_idx`, `kp_x`, `kp_y`, `conf`, `clip`) are fixed so that the tracking, body-language, and dashboard stages can all rely on them. The tracking stage emits a coverage-statistics JSON and image plots per clip; the body-language stage emits a per-second metrics table; the training stage emits the trained weights and a metrics summary. The integration stage reads all of these through one loader.

### Stages and Responsibilities

#### 1. Pose Inference

Loads a YOLO11 pose model (`yolo11m-pose`) and runs real inference on an image and on the lecture clips. It extracts the seventeen COCO keypoints for the presenter each frame and writes them to `presenter_keypoints.csv`. The privacy rule lives here: with several people visible per frame, only the largest, most central figure is kept, so the keypoints table never contains an audience member.

#### 2. Tracking, Heatmap, and Spatial Coverage

Runs a real OpenCV capture-process-write loop. It tracks the presenter with ByteTrack for a stable identity, draws the pose and a movement trail, and renders a movement heatmap with the Ultralytics `solutions.Heatmap` module. From the presenter's centroid path it computes a grid-occupancy percentage over a ten-by-ten room grid, a total distance travelled, and a movement-path plot, and writes the coverage statistics to a per-clip JSON.

#### 3. Body-Language Metrics

Consumes the keypoints table and derives three per-second signals. Posture stability comes from the torso angle between the shoulder and hip midpoints, expressed as a stability score. Gesture frequency comes from wrist-keypoint motion, counted as gesture events per minute. Movement energy comes from overall keypoint displacement per second. The result is `body_language_metrics.csv`, a per-second time series that drives the delivery timeline. Seconds where the torso is not confidently visible are left as gaps rather than filled in, so the signal stays honest.

#### 4. Custom Training and Evaluation

Fine-tunes a YOLO11 detector on a hand-gesture dataset across two documented runs: a baseline (full fine-tune, default augmentation) and a tuned run (frozen backbone, softened colour augmentation, added geometric jitter, lower learning rate) chosen in response to what the baseline showed. It then evaluates the winning weights on the held-out test split, reporting mean average precision, precision, recall, and a confusion matrix, and selects an operating point through a confidence and IoU threshold sweep.

#### 5. Integration, Report, and Export

Unifies the keypoints, coverage statistics, per-second metrics, and model summary into one dataset through a single loader. It exports the trained gesture model to ONNX, then renders the post-lecture analytics dashboard as a self-contained HTML report and serves the same data through an interactive Streamlit app.

---

## Workflow

```text
Lecture Recording  (single-presenter clips)
   |
   v
Pose Inference  (YOLO11m-pose, 17 COCO keypoints)
   |   privacy filter: presenter kept, audience dropped
   |   -> outputs/presenter_keypoints.csv
   |
   +---------------------------------+
   |                                 |
   v                                 v
Tracking, Heatmap & Coverage     Body-Language Metrics
(ByteTrack + OpenCV +            (posture / gesture / movement,
 solutions.Heatmap)              per second)
 -> coverage_stats.json,          -> body_language_metrics.csv
    heatmap + path plots

Custom Training & Evaluation  (independent branch)
(YOLO11n fine-tune, 2 runs -> val on held-out test split)
 -> gesture_detector_best.pt, gesture_metrics_summary.json
   |
   v
Integration, Report & Export
 - unify keypoints + coverage + metrics + model card
 - model.export -> ONNX
 - delivery-timeline analytics report (HTML) + Streamlit dashboard
```

Each arrow is a file hand-off: a stage writes an artifact to `outputs/`, and the next stage reads it.

---

## Pose Estimation and Privacy

Pose estimation is the core task beyond plain detection. A YOLO11 pose model extracts seventeen body keypoints per person per frame, and MentorVision keeps only the presenter's.

The privacy rule is implemented in code, not left as a note. Every clip has five to eleven people visible, so for each frame the pipeline selects the single largest bounding box (the figure closest to the camera and most central on stage) and discards the rest before any keypoint is recorded. The keypoints table that every later stage reads therefore contains the presenter alone, and no audience member is measured or stored at any point downstream.

---

## Tracking, Heatmap, and Spatial Coverage

This stage answers how the presenter used the room. ByteTrack gives the presenter a stable identity across frames inside a real OpenCV pipeline that reads, processes, and writes video. A movement heatmap, rendered with the Ultralytics solutions module, shows where time was spent, and an aggregate heatmap image is saved per clip.

Spatial coverage is quantified from the presenter's centroid path: a grid-occupancy percentage over a ten-by-ten room grid, a total distance travelled in pixels, and a movement-path plot with the start and end marked. Because clip resolutions differ, the stability score, grid occupancy, and gestures-per-minute are the metrics that compare fairly across clips, while pixel distance and movement energy are resolution-dependent and read within a clip rather than across clips.

---

## Body-Language Metrics

Three signals summarize physical delivery, all derived from the keypoints table and reported per second.

**Posture stability** is the torso angle between the shoulder midpoint and the hip midpoint, measured against vertical and converted to a stability score, so a steady upright posture scores high and swaying scores lower. **Gesture frequency** is detected from wrist-keypoint motion and reported as gesture events per minute, with a small per-second event count for the timeline. **Movement energy** is the overall per-second displacement of the presenter's keypoints, capturing how animated the delivery is.

Seconds in which the torso is not confidently visible are recorded as gaps rather than interpolated, so the posture signal reflects only what was actually measured.

---

## Custom Training and Evaluation

A YOLO11 detector is fine-tuned to recognize hand-gesture types, complementing the gesture-frequency metric with gesture identity.

**Two runs, with a reason for the second.** Run one is a baseline full fine-tune with default augmentation, whose job is to reveal how the small, domain-specific dataset behaves. Run two responds to that signal: it freezes the backbone to lean on pretrained features, softens the colour augmentation (hands are not colour-diagnostic), adds geometric jitter to mimic camera-angle variation, and lowers the learning rate for the frozen-backbone fine-tune.

**Evaluation on unseen data.** The winning weights are validated on the held-out test split, giving an honest read on generalization:

| Metric | Value |
|---|---|
| mAP@50 | 0.995 |
| mAP@50-95 | 0.582 |
| Precision | 0.915 |
| Recall | 0.943 |

**A justified operating point.** A confidence and IoU threshold sweep motivates the deployed point of confidence 0.25 and IoU 0.5. Because a missed gesture (which under-counts engagement in the report) costs more than an extra flagged one, the operating point is chosen to favour recall without letting precision fall below 0.90.

---

## Deployment and Export

The trained gesture model is exported to **ONNX** through `model.export`. ONNX is a portable, framework-agnostic format: the exported model runs under ONNX Runtime with no PyTorch dependency at serve time, which keeps deployment lightweight and CPU-friendly, matching a post-lecture analytics tool that should be cheap to run rather than a live GPU service.

The analytics itself is served two ways from the same integrated data. A **self-contained HTML report** embeds every image and chart inline, so it renders in Colab and opens anywhere without a server. An **interactive Streamlit dashboard** presents the same summary cards, delivery timeline, and spatial-coverage visuals clip by clip. Both read a single integration layer, so a number or a colour always means the same thing across the two surfaces.

---

## Datasets

**Presentation clips.** Three royalty-free single-presenter clips from Pexels (Pexels License), used for pose estimation, tracking, and body-language analysis. Sources and licenses are logged in `data/README.md` for traceability.

**Gesture dataset.** A hand-gesture recognition dataset from Roboflow Universe in YOLO format, with five classes (`hi`, `okay`, `peace`, `thumbs_down`, `thumbs_up`), used to fine-tune the gesture detector. Its source and license are logged with the training stage. The datasets themselves are not committed to the repository; the notebooks download or reference them.

---

## Technologies

- Python
- Google Colab
- Ultralytics YOLO11 (`yolo11m-pose`, `yolo11n`)
- ByteTrack (multi-object tracking)
- OpenCV
- Ultralytics `solutions.Heatmap`
- Roboflow (dataset)
- ONNX and ONNX Runtime (export target)
- Streamlit (interactive dashboard)
- pandas, matplotlib, Pillow

---

## Repository Structure

```text
Computer-Vision-Capstone-Project/
├── notebooks/
│   ├── 01_inference.ipynb                  # pose inference + presenter keypoints (privacy filter)
│   ├── 02_tracking_heatmap_coverage.ipynb  # ByteTrack + heatmap + spatial coverage
│   ├── 03_body_language_metrics.ipynb      # posture / gesture / movement, per second
│   ├── 04_custom_training_evaluation.ipynb # gesture detector fine-tune + evaluation
│   └── 05_dashboard_export.ipynb           # integration, ONNX export, report + dashboard
├── src/                                    # shared modules the notebooks import
│   ├── pose_utils.py                       # keypoint extraction + presenter privacy rule
│   ├── body_language_metrics.py            # posture / gesture / movement metric pipeline
│   ├── body_language_plots.py              # reusable metric plots
│   ├── dashboard_data.py                   # integration layer: unifies every stage's output
│   ├── report_builder.py                   # self-contained HTML report + delivery timeline
│   └── dashboard.py                        # interactive Streamlit app
├── outputs/                                # committed analysis artifacts (evidence)
│   ├── presenter_keypoints.csv             # per-frame presenter keypoints (the contract)
│   ├── body_language_metrics.csv           # per-second posture / gesture / movement
│   ├── clip{1,2,3}_presenter_coverage_stats.json
│   ├── clip{1,2,3}_presenter_aggregate_heatmap.png
│   ├── clip{1,2,3}_presenter_movement_path.png
│   ├── plot_posture_stability.png
│   ├── plot_gesture_frequency.png
│   ├── plot_movement_dynamics.png
│   ├── gesture_metrics_summary.json        # evaluation metrics + operating point
│   └── mentorvision_report.html            # post-lecture analytics report
├── data/
│   └── README.md                           # clip sources and licenses
├── models/                                 # trained weights + ONNX export (git-ignored)
├── .gitignore
└── README.md
```

The executed notebooks in `notebooks/` are the reproducible evidence for each stage. The `src/` package is the same code organized for import, and `outputs/` holds the committed artifacts the dashboard renders from.

---

## Prerequisites

- Python 3.11 or later
- Google Colab with a GPU runtime (recommended for the pose, tracking, and training stages)
- The three presentation clips (see `data/README.md`) placed in the session before running the pose, tracking, and body-language stages
- A Roboflow API key for the training stage (read from a Colab secret, not hard-coded)
- The trained weights `gesture_detector_best.pt` in `models/` for the export stage

MentorVision does **not** store API keys or model weights in the GitHub repository.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/GhalaAwd/Computer-Vision-Capstone-Project.git
cd Computer-Vision-Capstone-Project
```

Each notebook installs its own dependencies in its first cells (`ultralytics`, and for the final stage `streamlit` and `pillow`), so no separate install step is required for Colab.

---

## How to Run

The notebooks are designed to run in order in Google Colab. Each writes its artifacts to `outputs/`, which the later stages read.

### 1. Pose inference

Open `notebooks/01_inference.ipynb`, place the presentation clips in the session, and run all. It produces `outputs/presenter_keypoints.csv`.

### 2. Tracking, heatmap, and coverage

Open `notebooks/02_tracking_heatmap_coverage.ipynb` and run all. Set `CLIP_IDX` to process each clip. It produces the per-clip coverage JSON, heatmap, and movement-path plots.

### 3. Body-language metrics

Open `notebooks/03_body_language_metrics.ipynb` and run all. It reads the keypoints table and produces `outputs/body_language_metrics.csv` and the metric plots.

### 4. Custom training and evaluation

Open `notebooks/04_custom_training_evaluation.ipynb`, add your Roboflow API key to the Colab Secrets panel, and run all. It fine-tunes the gesture detector, evaluates it, and produces the trained weights and `gesture_metrics_summary.json`.

### 5. Integration, report, and export

Open `notebooks/05_dashboard_export.ipynb` and run all. Provide the trained weights in `models/` for the export cell (mount Drive or upload). It exports the model to ONNX, builds `outputs/mentorvision_report.html`, and can launch the Streamlit dashboard.

---

## Expected Output

MentorVision produces:

- A per-frame presenter keypoints table, filtered to the presenter alone
- Per-clip spatial coverage: grid occupancy, distance travelled, a movement heatmap, and a movement-path plot
- Per-second posture stability, gesture frequency, and movement energy
- A fine-tuned gesture detector with a full evaluation (mean average precision, precision, recall, confusion matrix) and a justified operating point
- The gesture model exported to ONNX
- A post-lecture analytics report with summary cards, a synced delivery timeline, and the spatial-coverage visuals, plus an interactive Streamlit dashboard of the same data

The executed notebooks additionally show the captured output for each stage: real inference logs, the OpenCV tracking and heatmap runs, the metrics validation report, the two training runs with their evaluation and threshold sweep, and the ONNX export.

---

## Capstone Concepts Demonstrated

MentorVision demonstrates the key concepts covered in **Computer Vision for Developers**, including:

- Real Ultralytics inference on images and video through the Python API
- A task beyond plain detection: YOLO11 pose estimation
- A real-world video-analytics solution: ByteTrack tracking and a movement heatmap inside an OpenCV pipeline
- Model evaluation with mean average precision, precision, recall, and a confusion matrix, interpreted for the use case
- Custom fine-tuning on a documented dataset across two deliberate runs
- Threshold selection justified by a measured confidence and IoU sweep
- Deployment through ONNX export and a served Streamlit application
- Executed notebooks with captured output as evidence for every stage
