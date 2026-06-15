from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_video_features(video_path: str | Path, max_samples: int = 48) -> dict[str, float]:
    """Extract inexpensive domain-routing features from a video.

    OpenCV and NumPy are optional package dependencies because routing from an
    already-created descriptor does not require them.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Direct video analysis requires opencv-python and numpy. "
            "Install them or pass a VideoDescriptor to the router."
        ) from exc

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

    sample_count = min(max_samples, frame_count) if frame_count > 0 else max_samples
    indices = (
        np.linspace(0, max(frame_count - 1, 0), sample_count, dtype=int)
        if frame_count > 0
        else np.arange(sample_count)
    )

    brightness_values: list[float] = []
    sharpness_values: list[float] = []
    motion_values: list[float] = []
    camera_motion_values: list[float] = []
    scene_changes = 0
    previous_gray: Any = None

    for frame_index in indices:
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        brightness_values.append(float(gray_small.mean() / 255.0))
        sharpness_values.append(float(cv2.Laplacian(gray_small, cv2.CV_64F).var()))

        if previous_gray is not None:
            diff = cv2.absdiff(gray_small, previous_gray)
            normalized_diff = float(diff.mean() / 255.0)
            motion_values.append(normalized_diff)
            if normalized_diff > 0.22:
                scene_changes += 1

            shift, _ = cv2.phaseCorrelate(
                previous_gray.astype(np.float32), gray_small.astype(np.float32)
            )
            diagonal = (160**2 + 90**2) ** 0.5
            camera_motion_values.append(min(((shift[0] ** 2 + shift[1] ** 2) ** 0.5) / diagonal, 1.0))

        previous_gray = gray_small

    capture.release()
    transitions = max(len(brightness_values) - 1, 1)
    return {
        "duration_s": float(duration),
        "fps": float(fps),
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else 0.0,
        "brightness": _mean(brightness_values),
        "brightness_std": _std(brightness_values),
        "sharpness": _mean(sharpness_values),
        "motion_intensity": _mean(motion_values),
        "camera_motion": _mean(camera_motion_values),
        "scene_change_rate": float(scene_changes / transitions),
        "sampled_frames": float(len(brightness_values)),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)
