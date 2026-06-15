from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .video_features import extract_video_features


@dataclass(frozen=True)
class VideoDescriptor:
    features: Mapping[str, float]
    context: str = ""
    specifications: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimilarityResult:
    dataset: str
    display_name: str
    similarity: float
    video_similarity: float
    context_similarity: float | None
    specification_similarity: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RouteDecision:
    selected_dataset: str
    selected_checkpoint: str
    decision_threshold: float
    calibrated_threshold: float | None
    routing_confidence: float
    similarity_kind: str
    similarities: tuple[SimilarityResult, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["similarities"] = [asdict(item) for item in self.similarities]
        return result


class ModelWeightRouter:
    """Select a checkpoint by estimated similarity to its training domain."""

    def __init__(
        self,
        model_root: str | Path,
        profiles_path: str | Path | None = None,
        validate_files: bool = True,
    ) -> None:
        self.model_root = Path(model_root)
        default_profiles = Path(__file__).with_name("profiles.json")
        self.profiles_path = Path(profiles_path) if profiles_path else default_profiles
        self.profiles: dict[str, dict[str, Any]] = json.loads(
            self.profiles_path.read_text(encoding="utf-8")
        )
        if validate_files:
            self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        for dataset, profile in self.profiles.items():
            checkpoint = self.model_root / profile["checkpoint"]
            metrics_path = self.model_root / profile["metrics"]
            if not checkpoint.is_file():
                errors.append(f"{dataset}: missing checkpoint {checkpoint}")
            if not metrics_path.is_file():
                errors.append(f"{dataset}: missing metrics {metrics_path}")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected = float(profile["decision_threshold"])
            actual = float(metrics["threshold"])
            if not math.isclose(expected, actual, abs_tol=1e-8):
                errors.append(
                    f"{dataset}: profile threshold {expected} does not match metrics {actual}"
                )
        if errors:
            raise ValueError("Invalid model router configuration:\n" + "\n".join(errors))

    def route_video(
        self,
        video_path: str | Path,
        context: str = "",
        specifications: Mapping[str, Any] | None = None,
        max_samples: int = 48,
    ) -> RouteDecision:
        features = extract_video_features(video_path, max_samples=max_samples)
        return self.route(
            VideoDescriptor(
                features=features,
                context=context,
                specifications=specifications or {},
            )
        )

    def route(self, descriptor: VideoDescriptor) -> RouteDecision:
        results = tuple(
            sorted(
                (
                    self._score_profile(dataset, profile, descriptor)
                    for dataset, profile in self.profiles.items()
                ),
                key=lambda result: result.similarity,
                reverse=True,
            )
        )
        selected = results[0]
        profile = self.profiles[selected.dataset]
        margin = selected.similarity - results[1].similarity if len(results) > 1 else selected.similarity
        routing_confidence = _clamp(0.65 * selected.similarity + 0.35 * margin)
        return RouteDecision(
            selected_dataset=selected.dataset,
            selected_checkpoint=str((self.model_root / profile["checkpoint"]).resolve()),
            decision_threshold=float(profile["decision_threshold"]),
            calibrated_threshold=_optional_float(profile.get("calibrated_threshold")),
            routing_confidence=round(routing_confidence, 4),
            similarity_kind="heuristic_domain_similarity",
            similarities=results,
        )

    def _score_profile(
        self,
        dataset: str,
        profile: Mapping[str, Any],
        descriptor: VideoDescriptor,
    ) -> SimilarityResult:
        feature_scores: list[tuple[str, float]] = []
        for name, target in profile["feature_profile"].items():
            if name not in descriptor.features:
                continue
            tolerance = max(float(target["tolerance"]), 1e-8)
            distance = abs(float(descriptor.features[name]) - float(target["center"]))
            feature_scores.append((name, math.exp(-0.5 * (distance / tolerance) ** 2)))

        video_score = _mean(score for _, score in feature_scores)
        context_score, matched_context = _keyword_score(
            descriptor.context,
            profile.get("context_keywords", []),
            self._context_keyword_frequency(),
        )
        specs_score, matched_specs = _specification_score(
            descriptor.specifications, profile.get("specification_tags", {})
        )

        weighted_scores = [(video_score, 0.60)]
        if context_score is not None:
            weighted_scores.append((context_score, 0.25))
        if specs_score is not None:
            weighted_scores.append((specs_score, 0.15))
        total_weight = sum(weight for _, weight in weighted_scores)
        similarity = sum(score * weight for score, weight in weighted_scores) / total_weight

        strongest_features = sorted(feature_scores, key=lambda item: item[1], reverse=True)[:3]
        reasons = [f"{name} fit {score:.0%}" for name, score in strongest_features]
        if matched_context:
            reasons.append("context matched: " + ", ".join(matched_context[:4]))
        if matched_specs:
            reasons.append("specifications matched: " + ", ".join(matched_specs[:4]))

        return SimilarityResult(
            dataset=dataset,
            display_name=str(profile["display_name"]),
            similarity=round(similarity, 4),
            video_similarity=round(video_score, 4),
            context_similarity=_round_optional(context_score),
            specification_similarity=_round_optional(specs_score),
            reasons=tuple(reasons),
        )

    def _context_keyword_frequency(self) -> dict[str, int]:
        frequency: dict[str, int] = {}
        for profile in self.profiles.values():
            for keyword in set(profile.get("context_keywords", [])):
                normalized = _normalize(keyword)
                frequency[normalized] = frequency.get(normalized, 0) + 1
        return frequency


def _keyword_score(
    text: str, keywords: list[str], keyword_frequency: Mapping[str, int]
) -> tuple[float | None, list[str]]:
    if not text.strip():
        return None, []
    normalized = _normalize(text)
    matched = [keyword for keyword in keywords if _normalize(keyword) in normalized]
    evidence = sum(1.0 / keyword_frequency[_normalize(keyword)] for keyword in matched)
    score = min(evidence / 2.5, 1.0) if matched else 0.10
    return score, matched


def _specification_score(
    specifications: Mapping[str, Any], tags: Mapping[str, list[str]]
) -> tuple[float | None, list[str]]:
    if not specifications:
        return None, []
    matched: list[str] = []
    category_scores: list[float] = []
    for category, value in specifications.items():
        expected = tags.get(str(category), [])
        if not expected:
            continue
        actual_values = value if isinstance(value, (list, tuple, set)) else [value]
        actual = {_normalize(str(item)) for item in actual_values}
        expected_normalized = {_normalize(item) for item in expected}
        overlap = actual & expected_normalized
        category_scores.append(1.0 if overlap else 0.0)
        matched.extend(f"{category}={item}" for item in sorted(overlap))
    return (_mean(category_scores), matched) if category_scores else (0.10, [])


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()


def _mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
