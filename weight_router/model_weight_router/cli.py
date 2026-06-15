from __future__ import annotations

import argparse
import json
from pathlib import Path

from .router import ModelWeightRouter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the GuardianEye violence-model checkpoint closest to an input video."
    )
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument(
        "--model-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Directory containing train_output_HF, train_output_NTU, and train_output_RLVS",
    )
    parser.add_argument("--context", default="", help="VLM/user context describing the video")
    parser.add_argument(
        "--specifications",
        default="{}",
        help='JSON object, for example: {"source":"cctv","camera":"fixed"}',
    )
    parser.add_argument("--max-samples", type=int, default=48)
    args = parser.parse_args()

    specifications = json.loads(args.specifications)
    if not isinstance(specifications, dict):
        parser.error("--specifications must be a JSON object")

    router = ModelWeightRouter(args.model_root)
    decision = router.route_video(
        args.video,
        context=args.context,
        specifications=specifications,
        max_samples=args.max_samples,
    )
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
