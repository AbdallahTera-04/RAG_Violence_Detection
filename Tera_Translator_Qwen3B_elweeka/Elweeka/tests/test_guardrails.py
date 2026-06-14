from rag_service.guardrails import validate_narrative


def assert_reason(result, expected):
    assert result.status == "regenerate_required"
    assert result.regeneration_instruction
    assert any(expected in reason for reason in result.reasons)


def test_guardrail_pass_case_with_supported_claims():
    result = validate_narrative(
        narrative=(
            "The classifier result is violence with confidence 0.9200. "
            "The explanation stays grounded in evidence describing two people pushing near a chair."
        ),
        verdict="violence",
        confidence=0.92,
        packet_summary="Evidence describes two people pushing.",
        retrieved_references=[
            {
                "reference_id": "ref-1",
                "source": "guideline",
                "snippet": "The scene includes a chair.",
            }
        ],
    )

    assert result.status == "passed"
    assert result.reasons == []
    assert result.regeneration_instruction is None


def test_verdict_contradiction_detection():
    result = validate_narrative(
        narrative="No violence occurred. The classifier confidence remains 0.9300.",
        verdict="violence",
        confidence=0.93,
        packet_summary="Evidence describes rapid motion.",
        retrieved_references=[],
    )

    assert_reason(result, "Verdict contradiction")


def test_confidence_modification_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.80. The packet is limited.",
        verdict="violence",
        confidence=0.94,
        packet_summary="Evidence describes rapid motion.",
        retrieved_references=[],
    )

    assert_reason(result, "Confidence modification")


def test_invented_weapon_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. A gun is visible.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes rapid motion.",
        retrieved_references=[],
    )

    assert_reason(result, "Invented weapon")


def test_invented_object_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. A bottle is visible.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes rapid motion.",
        retrieved_references=[],
    )

    assert_reason(result, "Invented object")


def test_invented_people_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. Three people are involved.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes two people near each other.",
        retrieved_references=[],
    )

    assert_reason(result, "Invented people")


def test_unsupported_action_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. One person slapped another.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes two people near each other.",
        retrieved_references=[],
    )

    assert_reason(result, "Unsupported claim")


def test_unsupported_exact_timing_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. The action happened exactly 3 seconds in.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes rapid motion without exact timing.",
        retrieved_references=[],
    )

    assert_reason(result, "Unsupported exact timing")


def test_unsupported_responsibility_detection():
    result = validate_narrative(
        narrative="The classifier result is violence with confidence 0.9100. The attacker is responsible.",
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes two people near each other.",
        retrieved_references=[],
    )

    assert_reason(result, "Unsupported person responsibility")


def test_references_can_support_objects_and_actions():
    result = validate_narrative(
        narrative=(
            "The classifier result is violence with confidence 0.9100. "
            "The explanation mentions two people pushing near a bottle."
        ),
        verdict="violence",
        confidence=0.91,
        packet_summary="Evidence describes two people.",
        retrieved_references=[
            {
                "reference_id": "ref-1",
                "source": "guideline",
                "snippet": "Reference snippet mentions pushing near a bottle.",
            }
        ],
    )

    assert result.status == "passed"
