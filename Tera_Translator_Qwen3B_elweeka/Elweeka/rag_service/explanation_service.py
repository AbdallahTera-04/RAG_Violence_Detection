"""Elweeka explanation service boundary.

This module wires Day 1 schemas, prompts, and guardrails together without
loading real VLM/LLM models. Future model integration should happen through
the injected narrator client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import ValidationError

from rag_service.guardrails import validate_narrative
from rag_service.prompts import format_narrator_prompt
from rag_service.schemas import (
    ExplanationInput,
    ExplanationOutput,
    GuardrailIssue,
    GuardrailResult,
)


class NarratorClient(Protocol):
    """Future VLM/LLM client interface."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        context: ExplanationInput,
        text_only: bool,
    ) -> str | dict[str, Any]:
        """Return a candidate narrative without changing classifier facts."""


def generate_clip_explanation(
    verdict: str,
    confidence: float,
    packet_summary: str,
    retrieved_references: list[Any],
    frames_ref: list[str] | None = None,
    narrator_client: NarratorClient | Callable[..., str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a guarded explanation result for one classifier output.

    This function does not retrieve references, build evidence packets, load
    models, or run real VLM/LLM inference.
    """

    explanation_input = ExplanationInput(
        verdict=verdict,
        confidence=confidence,
        packet_summary=packet_summary,
        retrieved_references=retrieved_references,
        frames_ref=frames_ref or [],
    )
    text_only = len(explanation_input.frames_ref) == 0

    messages = format_narrator_prompt(
        verdict=explanation_input.verdict,
        confidence=explanation_input.confidence,
        packet_summary=explanation_input.packet_summary,
        retrieved_references=explanation_input.retrieved_references,
        frames_ref=explanation_input.frames_ref,
    )

    if narrator_client is None:
        candidate_narrative = _build_mock_fallback_narrative(
            verdict=explanation_input.verdict,
            confidence=explanation_input.confidence,
            text_only=text_only,
        )
    else:
        candidate_narrative = _extract_narrative(
            _call_narrator_client(
                narrator_client=narrator_client,
                messages=messages,
                context=explanation_input,
                text_only=text_only,
            )
        )

    guardrail_result = validate_narrative(
        narrative=candidate_narrative,
        verdict=explanation_input.verdict,
        confidence=explanation_input.confidence,
        packet_summary=explanation_input.packet_summary,
        retrieved_references=explanation_input.retrieved_references,
    )

    if guardrail_result.status == "passed":
        output = ExplanationOutput(
            narrative=candidate_narrative,
            guardrail_status="passed",
            limitations_note=_build_limitations_note(
                text_only=text_only,
                has_references=bool(explanation_input.retrieved_references),
            ),
        )
    else:
        output = ExplanationOutput(
            narrative=_build_guarded_rejection_narrative(
                verdict=explanation_input.verdict,
                confidence=explanation_input.confidence,
            ),
            guardrail_status=guardrail_result.status,
            limitations_note=_build_regeneration_limitations_note(guardrail_result.reasons),
            guardrail_result=_to_schema_guardrail_result(guardrail_result),
        )

    return output.model_dump(exclude_none=True)


def _call_narrator_client(
    *,
    narrator_client: NarratorClient | Callable[..., str | dict[str, Any]],
    messages: list[dict[str, str]],
    context: ExplanationInput,
    text_only: bool,
) -> str | dict[str, Any]:
    generate = getattr(narrator_client, "generate", None)
    target = generate if callable(generate) else narrator_client

    try:
        return target(messages=messages, context=context, text_only=text_only)
    except TypeError:
        try:
            return target(messages=messages)
        except TypeError:
            return target(messages)


def _extract_narrative(response: str | dict[str, Any] | Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        narrative = response.get("narrative")
        if isinstance(narrative, str):
            return narrative.strip()

    narrative = getattr(response, "narrative", None)
    if isinstance(narrative, str):
        return narrative.strip()

    raise ValueError("narrator_client must return a narrative string or object")


def _build_mock_fallback_narrative(
    *,
    verdict: str,
    confidence: float,
    text_only: bool,
) -> str:
    evidence_scope = (
        "the supplied packet and retrieved references"
        if text_only
        else "the supplied packet, retrieved references, and stored frame references"
    )
    return (
        f"The classifier result is {verdict} with confidence {confidence:.4f}. "
        f"This fallback explanation is limited to {evidence_scope} and adds no unsupported scene details."
    )


def _build_guarded_rejection_narrative(*, verdict: str, confidence: float) -> str:
    return (
        "The candidate explanation was rejected by the guardrails. "
        f"The fixed classifier result remains {verdict} with confidence {confidence:.4f}."
    )


def _build_limitations_note(*, text_only: bool, has_references: bool) -> str:
    notes: list[str] = []
    if text_only:
        notes.append("Text-only fallback used because no frame references were provided.")
    else:
        notes.append("Explanation used provided frame references without loading real VLM models.")

    if not has_references:
        notes.append("No retrieved references were provided.")

    notes.append("Verdict and confidence were preserved as fixed classifier facts.")
    return " ".join(notes)


def _build_regeneration_limitations_note(reasons: list[str]) -> str:
    reason_text = "; ".join(reasons) if reasons else "unspecified guardrail issue"
    return (
        "Guardrails require regeneration before this explanation can be accepted. "
        "Verdict and confidence were preserved as fixed classifier facts. "
        f"Reasons: {reason_text}"
    )


def _to_schema_guardrail_result(guardrail_result: Any) -> GuardrailResult:
    return GuardrailResult(
        status=guardrail_result.status,
        issues=[
            GuardrailIssue(
                code=_reason_to_code(reason),
                message=reason,
                severity="error",
            )
            for reason in guardrail_result.reasons
        ],
        regeneration_instruction=guardrail_result.regeneration_instruction,
    )


def _reason_to_code(reason: str) -> str:
    code = reason.split(":", 1)[0].strip().lower().replace(" ", "_")
    return code or "guardrail_violation"


__all__ = ["generate_clip_explanation", "NarratorClient", "ValidationError"]
