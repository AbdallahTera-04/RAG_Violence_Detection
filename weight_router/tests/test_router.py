from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_weight_router import ModelWeightRouter, VideoDescriptor


class ModelWeightRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        package_profiles = Path(__file__).parents[1] / "model_weight_router" / "profiles.json"
        self.profiles = json.loads(package_profiles.read_text(encoding="utf-8"))
        for profile in self.profiles.values():
            checkpoint = self.root / profile["checkpoint"]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            metrics = self.root / profile["metrics"]
            metrics.write_text(
                json.dumps({"threshold": profile["decision_threshold"]}),
                encoding="utf-8",
            )
        self.router = ModelWeightRouter(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_hockey_context_routes_to_hf(self) -> None:
        decision = self.router.route(
            VideoDescriptor(
                features={
                    "duration_s": 2.0,
                    "fps": 25.0,
                    "aspect_ratio": 1.5,
                    "brightness": 0.65,
                    "motion_intensity": 0.12,
                    "camera_motion": 0.07,
                    "scene_change_rate": 0.04,
                },
                context="Two hockey players fight on the ice rink during a broadcast game.",
                specifications={"environment": "arena", "source": "broadcast"},
            )
        )
        self.assertEqual(decision.selected_dataset, "HF")
        self.assertAlmostEqual(decision.decision_threshold, 0.70)

    def test_fixed_surveillance_routes_to_ntu(self) -> None:
        decision = self.router.route(
            VideoDescriptor(
                features={
                    "duration_s": 8.0,
                    "fps": 20.0,
                    "aspect_ratio": 1.77,
                    "brightness": 0.40,
                    "motion_intensity": 0.05,
                    "camera_motion": 0.01,
                    "scene_change_rate": 0.0,
                },
                context="Fixed CCTV surveillance camera overlooking a public street.",
                specifications={"source": "cctv", "camera": "fixed"},
            )
        )
        self.assertEqual(decision.selected_dataset, "NTU")
        self.assertAlmostEqual(decision.decision_threshold, 0.08)

    def test_handheld_real_world_routes_to_rlvs(self) -> None:
        decision = self.router.route(
            VideoDescriptor(
                features={
                    "duration_s": 5.0,
                    "fps": 30.0,
                    "aspect_ratio": 1.77,
                    "brightness": 0.45,
                    "motion_intensity": 0.14,
                    "camera_motion": 0.13,
                    "scene_change_rate": 0.05,
                },
                context="A real life street fight recorded on a handheld phone.",
                specifications={"source": "phone", "camera": "handheld"},
            )
        )
        self.assertEqual(decision.selected_dataset, "RLVS")
        self.assertAlmostEqual(decision.decision_threshold, 0.21)

    def test_output_contains_ranked_similarity_for_every_dataset(self) -> None:
        decision = self.router.route(VideoDescriptor(features={"duration_s": 4.0}))
        self.assertEqual(len(decision.similarities), 3)
        scores = [item.similarity for item in decision.similarities]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(decision.similarity_kind, "heuristic_domain_similarity")

    def test_generic_violence_words_do_not_decide_the_dataset(self) -> None:
        baseline = self.router.route(VideoDescriptor(features={"duration_s": 5.0}))
        violent = self.router.route(
            VideoDescriptor(
                features={"duration_s": 5.0},
                context="A fight and violent assault occurs.",
            )
        )
        self.assertEqual(baseline.selected_dataset, violent.selected_dataset)


if __name__ == "__main__":
    unittest.main()
