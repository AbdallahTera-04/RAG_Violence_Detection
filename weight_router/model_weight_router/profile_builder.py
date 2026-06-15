from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .video_features import extract_video_features

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
ROUTING_FEATURES = (
    "duration_s",
    "fps",
    "aspect_ratio",
    "brightness",
    "motion_intensity",
    "camera_motion",
    "scene_change_rate",
)


def build_empirical_profile(
    dataset_root: str | Path, max_videos: int | None = None, max_samples: int = 48
) -> dict[str, object]:
    """Measure robust routing-feature centers from exact training videos."""
    root = Path(dataset_root)
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if max_videos:
        videos = videos[:max_videos]
    if not videos:
        raise ValueError(f"No supported videos found under {root}")

    rows: list[dict[str, float]] = []
    failures: list[str] = []
    for video in videos:
        try:
            rows.append(extract_video_features(video, max_samples=max_samples))
        except Exception as exc:
            failures.append(f"{video}: {exc}")
    if not rows:
        raise RuntimeError("No videos could be analyzed")

    profile: dict[str, dict[str, float]] = {}
    for feature in ROUTING_FEATURES:
        values = [row[feature] for row in rows]
        center = statistics.median(values)
        mad = statistics.median(abs(value - center) for value in values)
        tolerance = max(1.4826 * mad, _tolerance_floor(feature))
        profile[feature] = {
            "center": round(center, 6),
            "tolerance": round(tolerance, 6),
        }

    return {
        "dataset_root": str(root.resolve()),
        "analyzed_videos": len(rows),
        "failed_videos": len(failures),
        "feature_profile": profile,
        "failures": failures,
    }


def _tolerance_floor(feature: str) -> float:
    return {
        "duration_s": 0.5,
        "fps": 1.0,
        "aspect_ratio": 0.05,
        "brightness": 0.03,
        "motion_intensity": 0.01,
        "camera_motion": 0.01,
        "scene_change_rate": 0.01,
    }[feature]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a routing profile from an exact training-video directory."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-samples", type=int, default=48)
    args = parser.parse_args()
    result = build_empirical_profile(args.dataset_root, args.max_videos, args.max_samples)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
