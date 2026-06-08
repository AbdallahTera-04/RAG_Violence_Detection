"""Prompt templates for the Elweeka narrator layer.

This module only formats prompts. It does not call models, orchestrate RAG,
build evidence packets, or generate explanations.
"""

from __future__ import annotations

from typing import Any


SYSTEM_NARRATOR_PROMPT = """You are Elweeka, the explanation and guardrail narrator for Guardian Eye.

The classifier verdict is a fixed fact. The classifier confidence is a fixed fact. You must never re-decide whether the clip is violence or non-violence, never change the verdict, and never change the confidence.

Your only job is to narrate a grounded explanation using only:
1. the provided stored frame references,
2. the evidence packet summary,
3. the retrieved reference snippets.

Forbidden behavior:
- Do not re-decide violence vs non-violence.
- Do not change the verdict.
- Do not change the confidence.
- Do not invent people, identities, weapons, objects, locations, or actions.
- Do not infer unsupported intent, cause, blame, or exact timing.
- Do not claim frame-level details unless they are supported by the supplied frames, packet summary, or retrieved references.
- Do not add facts from outside the supplied context.

Output requirements:
- Write exactly one explanation paragraph.
- Use 2 to 4 grounded sentences.
- Treat heuristic person counts, timing, and motion descriptions cautiously.
- Mention limitations when the evidence is incomplete, ambiguous, text-only, or frame references are unavailable.
- The explanation must support the fixed verdict and confidence without replacing them."""


USER_NARRATOR_TEMPLATE = """Fixed classifier result:
- Verdict: {verdict}
- Confidence: {confidence}

Evidence packet summary:
{packet_summary}

Stored frame references:
{frames_ref}

Retrieved reference snippets:
{retrieved_references}

Write the final grounded explanation now. Remember: the verdict and confidence are immutable, and the answer must be 2 to 4 grounded sentences."""


def format_narrator_prompt(
    *,
    verdict: str,
    confidence: float,
    packet_summary: str,
    retrieved_references: list[Any],
    frames_ref: list[str],
) -> list[dict[str, str]]:
    """Return chat-style narrator messages ready for future model use.

    The formatter is intentionally lightweight so it can serve both VLM calls
    with frame references and text-only fallback calls with an empty frames list.
    """

    if len(frames_ref) > 16:
        raise ValueError("frames_ref supports at most 16 stored frames")

    user_prompt = USER_NARRATOR_TEMPLATE.format(
        verdict=verdict,
        confidence=f"{confidence:.4f}",
        packet_summary=packet_summary.strip(),
        retrieved_references=_format_retrieved_references(retrieved_references),
        frames_ref=_format_frames_ref(frames_ref),
    )
    return [
        {"role": "system", "content": SYSTEM_NARRATOR_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_frames_ref(frames_ref: list[str]) -> str:
    if not frames_ref:
        return "- No stored frames provided; use text-only fallback context."

    return "\n".join(
        f"- Frame {index:02d}: {frame_ref}"
        for index, frame_ref in enumerate(frames_ref, start=1)
    )


def _format_retrieved_references(retrieved_references: list[Any]) -> str:
    if not retrieved_references:
        return "- No retrieved reference snippets provided."

    return "\n".join(
        f"- Reference {index}: {_format_reference(reference)}"
        for index, reference in enumerate(retrieved_references, start=1)
    )


def _format_reference(reference: Any) -> str:
    if hasattr(reference, "model_dump"):
        reference = reference.model_dump(exclude_none=True)

    if isinstance(reference, dict):
        source = reference.get("source") or reference.get("reference_id") or "unknown"
        snippet = reference.get("snippet") or reference.get("summary") or reference
        return f"{source}: {snippet}"

    return str(reference)
