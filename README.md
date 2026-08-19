# Computer-Vision-Capstone-Project
## Pipeline Foundation & Pose Inference (Member 1)

### What this does
Loads a YOLO pose-estimation model (`yolo11m-pose.pt`) and runs real
inference on images and video via the Ultralytics Python API. Extracts
per-frame COCO-17 body keypoints for the main presenter in a video,
filtering out audience members for privacy.

### Pipeline overview
1. `model = YOLO("yolo11m-pose.pt")` loads pretrained pose weights
2. `model.predict()` runs inference on a test image and 3 presentation
   video clips (see `data/README.md` for clip sources/licenses)
3. `src/pose_utils.py` wraps this into `extract_presenter_keypoints()`,
   which processes a video frame-by-frame (`stream=True`, to avoid
   holding all results in RAM) and returns a keypoints DataFrame

### The CSV contract
`extract_presenter_keypoints()` returns columns:

| Column      | Meaning                                          |
|-------------|---------------------------------------------------|
| `frame`     | Frame index in the video                          |
| `person_id` | Fixed as `"presenter"` (single presenter per video)|
| `kp_idx`    | COCO-17 keypoint index (0=nose, 5=left_shoulder, 9=left_wrist, etc. — full list in `pose_utils.py` docstring) |
| `kp_x`      | Keypoint x-coordinate (pixels)                    |
| `kp_y`      | Keypoint y-coordinate (pixels)                    |
| `conf`      | Model confidence for that keypoint (0-1)           |

**This schema is the contract M2 and M3 build on** — column names
should not change without team agreement. Confidence is not
pre-filtered; downstream consumers should apply their own `conf`
threshold as needed.

### Privacy rule
Each clip has 5-11 people visible per frame (audience included). Only
the presenter is kept, identified as the person with the largest
bounding-box area per frame (closest to camera / most central on
stage). This filtering happens in code (`pose_utils.py`), not as a
manual step.

### Evidence
- `notebooks/01_inference.ipynb` — executed notebook with captured
  output for image inference, video inference (3 clips), and the
  full keypoint-extraction run
- `outputs/presenter_keypoints.csv` — sample keypoints CSV, all 3 clips
