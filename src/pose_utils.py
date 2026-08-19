"""
pose_utils.py

Core contract for the MentorVision pipeline. Runs YOLO pose estimation
over a video and returns per-frame COCO-17 keypoints as a DataFrame,
filtered to the main presenter only (privacy rule -- audience dropped).

Everyone else on the team (M2, M3) builds on top of the CSV this
produces, so don't change the column names without telling the team.

COCO-17 keypoint index reference (used by the kp_idx column):
 0 nose        1 left_eye     2 right_eye    3 left_ear     4 right_ear
 5 left_shoulder  6 right_shoulder  7 left_elbow   8 right_elbow
 9 left_wrist     10 right_wrist    11 left_hip     12 right_hip
13 left_knee      14 right_knee     15 left_ankle   16 right_ankle
"""

import pandas as pd
from ultralytics import YOLO


def extract_presenter_keypoints(video_path: str, model_path: str = "yolo11m-pose.pt") -> pd.DataFrame:
    """
    Run pose estimation over a video and return per-frame keypoints
    for the main presenter only (audience filtered out).

    Uses stream=True so results are processed one frame at a time
    instead of held in RAM all at once -- without this, longer videos
    or crowded clips (many people per frame) can crash the runtime.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    model_path : str
        Path or name of the YOLO pose model weights.

    Returns
    -------
    pd.DataFrame
        Columns: frame, person_id, kp_idx, kp_x, kp_y, conf
        One row per keypoint per frame, presenter only.
    """
    model = YOLO(model_path)

    results_generator = model.predict(
        source=video_path,
        stream=True,
    )

    rows = []
    for frame_idx, result in enumerate(results_generator):
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            continue

        # --- Privacy rule: keep only the main presenter ---
        # Heuristic: the presenter is the person with the largest
        # bounding-box area (closest to camera / most central on stage).
        # Your clips have 5-11 people per frame (audience visible), so
        # this filter is essential -- without it the CSV would mix in
        # random audience members.
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] -> x1,y1,x2,y2
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        presenter_idx = int(areas.argmax())

        person_id = "presenter"  # single fixed label per video

        kpts = result.keypoints.xy[presenter_idx].cpu().numpy()   # [17, 2]
        confs = result.keypoints.conf[presenter_idx].cpu().numpy()  # [17]

        for kp_idx in range(kpts.shape[0]):
            rows.append({
                "frame": frame_idx,
                "person_id": person_id,
                "kp_idx": kp_idx,
                "kp_x": float(kpts[kp_idx, 0]),
                "kp_y": float(kpts[kp_idx, 1]),
                "conf": float(confs[kp_idx]),
            })

    df = pd.DataFrame(rows, columns=["frame", "person_id", "kp_idx", "kp_x", "kp_y", "conf"])
    return df


if __name__ == "__main__":
    # Quick manual test on all 3 clips -- adjust paths if needed.
    clips = [
        "/content/clip1_presenter.mp4",
        "/content/clip2_presenter.mp4",
        "/content/clip3_presenter.mp4",
    ]

    all_dfs = []
    for clip_path in clips:
        clip_name = clip_path.split("/")[-1].replace(".mp4", "")
        print(f"Processing {clip_name}...")
        df = extract_presenter_keypoints(clip_path)
        df["clip"] = clip_name  # tag which clip each row came from
        all_dfs.append(df)
        print(f"  -> {len(df)} keypoint rows ({df['frame'].nunique()} frames)")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("outputs/presenter_keypoints.csv", index=False)
    print(f"\nSaved {len(combined)} total keypoint rows to outputs/presenter_keypoints.csv")
    print(combined.head(20))
