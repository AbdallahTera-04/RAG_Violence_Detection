"""Deterministic guardrails for Elweeka explanation narratives.

This module validates generated text after a future narrator model call. It
does not call a VLM/LLM, rewrite narratives, or alter classifier outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field


GuardrailStatus = Literal["passed", "failed", "regenerate_required"]


class GuardrailResult(BaseModel):
    """Result returned by deterministic narrative validation."""

    model_config = ConfigDict(extra="forbid")

    status: GuardrailStatus
    reasons: list[str] = Field(default_factory=list)
    regeneration_instruction: str | None = None


WEAPON_TERMS = {
    "axe",
    "bat",
    "blade",
    "club",
    "firearm",
    "gun",
    "hammer",
    "knife",
    "machete",
    "pistol",
    "rifle",
    "shotgun",
    "stick",
    "sword",
    "weapon",
}

OBJECT_TERMS = {
    "bag",
    "bottle",
    "box",
    "car",
    "chair",
    "door",
    "glass",
    "phone",
    "rock",
    "table",
    "vehicle",
    "window",
}

PERSON_TERMS = {
    "adult",
    "attacker",
    "bystander",
    "child",
    "crowd",
    "guard",
    "individual",
    "man",
    "men",
    "officer",
    "people",
    "person",
    "persons",
    "suspect",
    "victim",
    "woman",
    "women",
}

ACTION_TERMS = {
    "assault",
    "attack",
    "bleeding",
    "carrying",
    "chasing",
    "falling",
    "fight",
    "fighting",
    "grabbing",
    "hitting",
    "holding",
    "kicking",
    "punching",
    "pushing",
    "shooting",
    "shoving",
    "slapping",
    "stabbing",
    "striking",
    "throwing",
    "threatening",
}

ACTION_ALIASES = {
    "assault": {"assault", "assaulted", "assaulting", "attack", "attacked", "attacking"},
    "attack": {"attack", "attacked", "attacking", "assault", "assaulted", "assaulting"},
    "bleeding": {"bleed", "bleeding", "blood"},
    "carrying": {"carry", "carried", "carrying"},
    "chasing": {"chase", "chased", "chasing"},
    "falling": {"fall", "fell", "falling"},
    "fight": {"fight", "fighting"},
    "fighting": {"fight", "fighting"},
    "grabbing": {"grab", "grabbed", "grabbing"},
    "hitting": {"hit", "hits", "hitting"},
    "holding": {"hold", "held", "holding"},
    "kicking": {"kick", "kicked", "kicking"},
    "punching": {"punch", "punched", "punching"},
    "pushing": {"push", "pushed", "pushing"},
    "shooting": {"shoot", "shot", "shooting"},
    "shoving": {"shove", "shoved", "shoving"},
    "slapping": {"slap", "slapped", "slapping"},
    "stabbing": {"stab", "stabbed", "stabbing"},
    "striking": {"strike", "struck", "striking"},
    "throwing": {"throw", "threw", "throwing"},
    "threatening": {"threat", "threaten", "threatened", "threatening"},
}

RESPONSIBILITY_TERMS = {
    "aggressor",
    "assailant",
    "attacker",
    "blame",
    "blamed",
    "caused",
    "fault",
    "instigated",
    "perpetrator",
    "responsible",
    "started",
    "suspect",
    "victim",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "sixteen": 16,
}


def validate_narrative(
    *,
    narrative: str,
    verdict: str,
    confidence: float,
    packet_summary: str,
    retrieved_references: list[Any],
    allowed_objects: Sequence[str] | None = None,
    allowed_weapons: Sequence[str] | None = None,
    allowed_people_count: int | None = None,
) -> GuardrailResult:
    """Validate a generated explanation without changing classifier facts."""

    reasons: list[str] = []
    narrative_text = narrative.strip()
    narrative_norm = _normalize(narrative_text)
    packet_norm = _normalize(packet_summary)
    reference_norm = _normalize(_references_to_text(retrieved_references))
    evidence_norm = f"{packet_norm} {reference_norm}".strip()

    reasons.extend(_check_verdict_contradiction(narrative_norm, verdict))
    reasons.extend(_check_confidence_modification(narrative_norm, confidence))
    reasons.extend(
        _check_disallowed_terms(
            narrative_norm=narrative_norm,
            evidence_norm=evidence_norm,
            allowed_terms=allowed_weapons,
            term_set=WEAPON_TERMS,
            reason_prefix="Invented weapon",
        )
    )
    reasons.extend(
        _check_disallowed_terms(
            narrative_norm=narrative_norm,
            evidence_norm=evidence_norm,
            allowed_terms=allowed_objects,
            term_set=OBJECT_TERMS,
            reason_prefix="Invented object",
        )
    )
    reasons.extend(
        _check_people_claims(
            narrative_norm=narrative_norm,
            evidence_norm=evidence_norm,
            allowed_people_count=allowed_people_count,
        )
    )
    reasons.extend(_check_unsupported_actions(narrative_norm, evidence_norm))
    reasons.extend(_check_exact_timing(narrative_norm, evidence_norm))
    reasons.extend(_check_person_responsibility(narrative_norm, evidence_norm))

    if reasons:
        return GuardrailResult(
            status="regenerate_required",
            reasons=_dedupe(reasons),
            regeneration_instruction=build_regeneration_instruction(
                verdict=verdict,
                confidence=confidence,
                reasons=reasons,
            ),
        )

    return GuardrailResult(status="passed")


def build_regeneration_instruction(
    *,
    verdict: str,
    confidence: float,
    reasons: Sequence[str] | None = None,
) -> str:
    """Build a rewrite instruction for a future narrator model call."""

    reason_text = ""
    if reasons:
        reason_text = " Rejection reasons: " + "; ".join(_dedupe(reasons))

    return (
        "Rewrite the explanation in 2 to 4 grounded sentences using only the "
        "fixed verdict, fixed confidence, packet_summary, retrieved_references, "
        "and frames_ref. Do not change or reinterpret the verdict "
        f"({verdict}) or confidence ({confidence:.4f}). Do not invent people, "
        "weapons, objects, actions, exact timing, or responsibility claims. "
        "Mention limitations when support is incomplete or ambiguous."
        f"{reason_text}"
    )


def _check_verdict_contradiction(narrative_norm: str, verdict: str) -> list[str]:
    if verdict == "violence":
        contradiction_patterns = [
            r"\bno violence (?:occurred|is shown|was detected|was observed)\b",
            r"\bno violent (?:act|action|behavior|incident) (?:occurred|is shown|was observed)\b",
            r"\b(?:the )?clip (?:is|was|appears) non violence\b",
            r"\b(?:the )?clip (?:is|was|appears) non violent\b",
            r"\bdoes not show violence\b",
            r"\bdid not show violence\b",
        ]
        if any(re.search(pattern, narrative_norm) for pattern in contradiction_patterns):
            return ["Verdict contradiction: classifier verdict is violence, but narrative says no violence occurred."]

    if verdict == "non-violence":
        violence_claims = [
            r"\bviolence occurred\b",
            r"\bviolent (?:act|action|behavior|incident) occurred\b",
            r"\b(?:an )?assault occurred\b",
            r"\b(?:a )?fight occurred\b",
            r"\bthe clip shows violence\b",
            r"\bthe scene is violent\b",
        ]
        if any(re.search(pattern, narrative_norm) for pattern in violence_claims):
            return ["Verdict contradiction: classifier verdict is non-violence, but narrative claims violence occurred."]

    return []


def _check_confidence_modification(narrative_norm: str, confidence: float) -> list[str]:
    reasons: list[str] = []
    confidence_windows = re.finditer(
        r"\b(?:confidence|probability|score)\b.{0,40}?(?P<number>\d+(?:\.\d+)?%?)",
        narrative_norm,
    )
    for match in confidence_windows:
        stated = _parse_confidence_number(match.group("number"))
        if stated is not None and abs(stated - confidence) > 0.005:
            reasons.append(
                "Confidence modification: narrative states "
                f"{match.group('number')} but classifier confidence is {confidence:.4f}."
            )

    if re.search(r"\b(?:lower|higher|reduced|increased|changed|adjusted) confidence\b", narrative_norm):
        reasons.append("Confidence modification: narrative describes changing the fixed confidence.")

    return reasons


def _check_disallowed_terms(
    *,
    narrative_norm: str,
    evidence_norm: str,
    allowed_terms: Sequence[str] | None,
    term_set: set[str],
    reason_prefix: str,
) -> list[str]:
    allowed_norm = _normalize(" ".join(allowed_terms or []))
    reasons = []
    for term in sorted(term_set):
        if _contains_word(narrative_norm, term):
            if _contains_word(evidence_norm, term) or _contains_word(allowed_norm, term):
                continue
            reasons.append(f"{reason_prefix}: '{term}' is not supported by packet_summary, references, or allowed lists.")
    return reasons


def _check_people_claims(
    *,
    narrative_norm: str,
    evidence_norm: str,
    allowed_people_count: int | None,
) -> list[str]:
    reasons: list[str] = []
    narrative_mentions_people = any(_contains_word(narrative_norm, term) for term in PERSON_TERMS)
    evidence_mentions_people = any(_contains_word(evidence_norm, term) for term in PERSON_TERMS)

    if narrative_mentions_people and not evidence_mentions_people and allowed_people_count is None:
        reasons.append("Invented people: narrative mentions people but packet_summary or references do not support people claims.")

    claimed_count = _extract_people_count(narrative_norm)
    evidence_count = _extract_people_count(evidence_norm)
    if (
        allowed_people_count is not None
        and claimed_count is not None
        and claimed_count > allowed_people_count
    ):
        reasons.append(
            "Invented people: narrative claims "
            f"{claimed_count} people but only {allowed_people_count} are allowed."
        )
    elif (
        allowed_people_count is None
        and claimed_count is not None
        and evidence_count is not None
        and claimed_count > evidence_count
    ):
        reasons.append(
            "Invented people: narrative claims "
            f"{claimed_count} people but evidence supports {evidence_count}."
        )

    for role in sorted(PERSON_TERMS - {"individual", "people", "person", "persons"}):
        if _contains_word(narrative_norm, role) and not _contains_word(evidence_norm, role):
            reasons.append(f"Invented people: role '{role}' is not supported by packet_summary or references.")

    return reasons


def _check_unsupported_actions(narrative_norm: str, evidence_norm: str) -> list[str]:
    reasons = []
    for action in sorted(ACTION_TERMS):
        aliases = ACTION_ALIASES.get(action, {action})
        narrative_has_action = any(_contains_word(narrative_norm, alias) for alias in aliases)
        evidence_has_action = any(_contains_word(evidence_norm, alias) for alias in aliases)
        if narrative_has_action and not evidence_has_action:
            reasons.append(f"Unsupported claim: action '{action}' is not present in packet_summary or references.")
    return reasons


def _check_exact_timing(narrative_norm: str, evidence_norm: str) -> list[str]:
    reasons = []
    timing_patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|frames?)\b",
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\b(?:at|after|before)\s+\d+(?:\.\d+)?\b",
        r"\bexactly\s+\d+(?:\.\d+)?\b",
    ]
    claims = _dedupe(
        match.group(0)
        for pattern in timing_patterns
        for match in re.finditer(pattern, narrative_norm)
    )
    for claim in claims:
        if claim not in evidence_norm:
            reasons.append(f"Unsupported exact timing: '{claim}' is not supported by packet_summary or references.")
    return reasons


def _check_person_responsibility(narrative_norm: str, evidence_norm: str) -> list[str]:
    reasons = []
    for term in sorted(RESPONSIBILITY_TERMS):
        if _contains_word(narrative_norm, term) and not _contains_word(evidence_norm, term):
            reasons.append(
                "Unsupported person responsibility: "
                f"'{term}' is not supported by packet_summary or references."
            )
    return reasons


def _extract_people_count(text: str) -> int | None:
    count_patterns = [
        r"\b(?P<count>\d+)\s+(?:people|persons|individuals|men|women|adults|children)\b",
        r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)\s+"
        r"(?:people|persons|individuals|men|women|adults|children)\b",
    ]
    for pattern in count_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        count = match.group("count")
        if count.isdigit():
            return int(count)
        return NUMBER_WORDS[count]
    return None


def _parse_confidence_number(value: str) -> float | None:
    value = value.strip()
    try:
        if value.endswith("%"):
            return float(value[:-1]) / 100
        parsed = float(value)
        if parsed > 1:
            return parsed / 100
        return parsed
    except ValueError:
        return None


def _references_to_text(retrieved_references: list[Any]) -> str:
    parts: list[str] = []
    for reference in retrieved_references:
        if hasattr(reference, "model_dump"):
            reference = reference.model_dump(exclude_none=True)
        if isinstance(reference, dict):
            parts.append(json.dumps(reference, sort_keys=True))
        else:
            parts.append(str(reference))
    return " ".join(parts)


def _contains_word(text: str, term: str) -> bool:
    if term == "knife":
        pattern = r"\b(?:knife|knives)\b"
    elif term.endswith("s"):
        pattern = rf"\b{re.escape(term)}\b"
    else:
        pattern = rf"\b{re.escape(term)}(?:s|es)?\b"
    return re.search(pattern, text) is not None


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower().replace("-", " ")).strip()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
