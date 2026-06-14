import json
from datetime import datetime
from pathlib import Path

from explanation_service import generate_clip_explanation
from rag_service.historical_questions import answer_historical_question
from rag_service.prompts import SYSTEM_NARRATOR_PROMPT, USER_NARRATOR_TEMPLATE


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "day5_mock_examples.json"
NOW = datetime(2026, 6, 9, 12, 0, 0)


class StaticNarratorClient:
    def __init__(self, narrative):
        self.narrative = narrative

    def generate(self, *, messages, context, text_only):
        return self.narrative


def load_examples():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_day5_saved_mock_examples_cover_explain_and_ask_flows():
    examples = load_examples()

    assert len(examples) == 4
    assert {example["kind"] for example in examples} == {"explain", "ask"}


def test_day5_e2e_saved_mock_examples_pass_contracts():
    for example in load_examples():
        if example["kind"] == "explain":
            input_data = dict(example["input"])
            mock_narrative = input_data.pop("mock_narrative", None)
            narrator_client = StaticNarratorClient(mock_narrative) if mock_narrative else None

            result = generate_clip_explanation(
                **input_data,
                narrator_client=narrator_client,
            )

            assert result["guardrail_status"] == example["expected_status"]
            assert result["narrative"]
            assert result["limitations_note"]
            assert str(input_data["confidence"])[:4] in result["narrative"]

        if example["kind"] == "ask":
            result = answer_historical_question(
                example["question"],
                records=example["records"],
                now=NOW,
            )

            assert result["matched_incidents"] == example["expected_matched_incidents"]
            assert result["answer"]
            assert result["limitations_note"]
            assert result["filters"]["verdict"] == "violence"


def test_day5_prompt_templates_are_finalized_for_immutability_and_grounding():
    assert "fixed fact" in SYSTEM_NARRATOR_PROMPT
    assert "never change the verdict" in SYSTEM_NARRATOR_PROMPT
    assert "never change the confidence" in SYSTEM_NARRATOR_PROMPT
    assert "2 to 4 grounded sentences" in USER_NARRATOR_TEMPLATE
    assert "{verdict}" in USER_NARRATOR_TEMPLATE
    assert "{confidence}" in USER_NARRATOR_TEMPLATE
    assert "{packet_summary}" in USER_NARRATOR_TEMPLATE
    assert "{retrieved_references}" in USER_NARRATOR_TEMPLATE
    assert "{frames_ref}" in USER_NARRATOR_TEMPLATE
