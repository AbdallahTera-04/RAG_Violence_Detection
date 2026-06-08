import sys

import pytest
from pydantic import ValidationError

from explanation_service import generate_clip_explanation


REFERENCE = {
    "reference_id": "ref-1",
    "source": "guardian-eye-guideline",
    "snippet": "Use cautious language and preserve classifier outputs.",
}


class StaticNarratorClient:
    def __init__(self, narrative):
        self.narrative = narrative
        self.calls = []

    def generate(self, *, messages, context, text_only):
        self.calls.append(
            {
                "messages": messages,
                "context": context,
                "text_only": text_only,
            }
        )
        return self.narrative


def test_normal_flow_uses_mock_fallback_and_passes_guardrails():
    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.94,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[REFERENCE],
        frames_ref=["frame_01.jpg", "frame_02.jpg"],
    )

    assert result["guardrail_status"] == "passed"
    assert "violence" in result["narrative"]
    assert "0.9400" in result["narrative"]
    assert "Verdict and confidence were preserved" in result["limitations_note"]


def test_guardrail_pass_with_injected_narrator_client():
    client = StaticNarratorClient(
        "The classifier result is violence with confidence 0.9100. "
        "The explanation stays grounded in the provided packet and frame references."
    )

    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.91,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[REFERENCE],
        frames_ref=["frame_01.jpg"],
        narrator_client=client,
    )

    assert result["guardrail_status"] == "passed"
    assert len(client.calls) == 1
    assert client.calls[0]["text_only"] is False
    assert client.calls[0]["context"].verdict == "violence"


def test_guardrail_regenerate_required_for_unsafe_narrative():
    client = StaticNarratorClient(
        "No violence occurred. The classifier confidence is 0.80."
    )

    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.94,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[REFERENCE],
        frames_ref=["frame_01.jpg"],
        narrator_client=client,
    )

    assert result["guardrail_status"] == "regenerate_required"
    assert "candidate explanation was rejected" in result["narrative"]
    assert "0.9400" in result["narrative"]
    assert "Guardrails require regeneration" in result["limitations_note"]
    assert "Verdict contradiction" in result["limitations_note"]
    assert "Confidence modification" in result["limitations_note"]
    assert result["guardrail_result"]["status"] == "regenerate_required"
    assert result["guardrail_result"]["regeneration_instruction"]
    assert result["guardrail_result"]["issues"]


def test_missing_frames_uses_text_only_fallback():
    result = generate_clip_explanation(
        verdict="non-violence",
        confidence=0.73,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[REFERENCE],
        frames_ref=None,
    )

    assert result["guardrail_status"] == "passed"
    assert "Text-only fallback used" in result["limitations_note"]
    assert "non-violence" in result["narrative"]
    assert "0.7300" in result["narrative"]


def test_empty_references_are_supported_with_limitation_note():
    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.86,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[],
        frames_ref=[],
    )

    assert result["guardrail_status"] == "passed"
    assert "No retrieved references were provided" in result["limitations_note"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "verdict": "unknown",
            "confidence": 0.5,
            "packet_summary": "Packet.",
            "retrieved_references": [],
        },
        {
            "verdict": "violence",
            "confidence": 1.2,
            "packet_summary": "Packet.",
            "retrieved_references": [],
        },
        {
            "verdict": "violence",
            "confidence": 0.5,
            "packet_summary": "   ",
            "retrieved_references": [],
        },
        {
            "verdict": "violence",
            "confidence": 0.5,
            "packet_summary": "Packet.",
            "retrieved_references": [],
            "frames_ref": [f"frame_{index:02d}.jpg" for index in range(17)],
        },
    ],
)
def test_schema_validation_errors(kwargs):
    with pytest.raises(ValidationError):
        generate_clip_explanation(**kwargs)


def test_verdict_immutability_when_client_contradicts_classifier():
    client = StaticNarratorClient(
        "No violence occurred. The explanation tries to reverse the fixed result."
    )

    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.88,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[],
        frames_ref=[],
        narrator_client=client,
    )

    assert result["guardrail_status"] == "regenerate_required"
    assert "violence" in result["narrative"]
    assert "reverse" not in result["narrative"]


def test_confidence_immutability_when_client_changes_confidence():
    client = StaticNarratorClient(
        "The classifier result is violence with confidence 0.40. "
        "That changed score should not be accepted."
    )

    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.88,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[],
        frames_ref=[],
        narrator_client=client,
    )

    assert result["guardrail_status"] == "regenerate_required"
    assert "0.8800" in result["narrative"]
    assert "0.40" not in result["narrative"]


def test_service_does_not_load_heavy_model_modules():
    sys.modules.pop("torch", None)
    sys.modules.pop("transformers", None)

    result = generate_clip_explanation(
        verdict="violence",
        confidence=0.81,
        packet_summary="Classifier packet is available for grounding.",
        retrieved_references=[],
        frames_ref=[],
    )

    assert result["guardrail_status"] == "passed"
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
