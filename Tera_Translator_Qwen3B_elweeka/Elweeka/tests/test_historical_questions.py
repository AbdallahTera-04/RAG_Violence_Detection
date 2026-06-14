from datetime import datetime, timedelta
import sys

from rag_service.historical_questions import (
    answer_historical_question,
    format_explain_response,
    parse_historical_filters,
    search_incidents,
    summarize_incidents,
)


NOW = datetime(2026, 6, 9, 12, 0, 0)


def incident(
    incident_id,
    *,
    verdict="violence",
    confidence=0.9,
    packet_summary="Stored packet summary.",
    narrative="Stored narrative.",
    timestamp=None,
    weapons=None,
):
    return {
        "incident_id": incident_id,
        "verdict": verdict,
        "confidence": confidence,
        "packet_summary": packet_summary,
        "narrative": narrative,
        "timestamp": timestamp,
        "weapons": weapons or [],
    }


def test_parse_historical_filters_for_verdict_weapon_and_time():
    filters = parse_historical_filters(
        "Tell me about violent incidents with a knife last week",
        now=NOW,
    )

    assert filters.verdict == "violence"
    assert filters.weapons == ["knife"]
    assert filters.start_time == NOW - timedelta(days=7)
    assert filters.end_time == NOW


def test_search_incidents_filters_before_vector_ranking():
    records = [
        incident(
            "old-knife",
            packet_summary="One person holding a knife.",
            narrative="Stored narrative mentions knife.",
            timestamp=NOW - timedelta(days=3),
            weapons=["knife"],
        ),
        incident(
            "non-violence-knife",
            verdict="non-violence",
            packet_summary="A knife is present in a non-violent scene.",
            narrative="Stored narrative mentions no contact.",
            timestamp=NOW - timedelta(days=2),
            weapons=["knife"],
        ),
        incident(
            "violence-bottle",
            packet_summary="One person holding a bottle.",
            narrative="Stored narrative mentions bottle.",
            timestamp=NOW - timedelta(days=1),
            weapons=["bottle"],
        ),
    ]
    seen_by_ranker = []

    def ranker(question, candidates, limit):
        seen_by_ranker.extend(record.incident_id for record in candidates)
        return list(reversed(candidates))[:limit]

    result = search_incidents(
        "violent knife incident last week",
        records=records,
        vector_ranker=ranker,
        now=NOW,
    )

    assert seen_by_ranker == ["old-knife"]
    assert [record.incident_id for record in result.records] == ["old-knife"]
    assert result.limitations_note


def test_search_incidents_can_use_injected_store():
    class Store:
        def __init__(self):
            self.filters = None

        def search(self, filters):
            self.filters = filters
            return [
                incident(
                    "store-hit",
                    packet_summary="Two people pushing.",
                    narrative="Stored narrative says pushing occurred.",
                    timestamp=NOW,
                )
            ]

    store = Store()
    result = search_incidents("violent incidents today", incident_store=store, now=NOW)

    assert store.filters.verdict == "violence"
    assert [record.incident_id for record in result.records] == ["store-hit"]


def test_summarize_incidents_uses_stored_packet_and_narrative_only():
    records = [
        incident(
            "inc-1",
            confidence=0.91,
            packet_summary="Packet says two people were close together.",
            narrative="Stored narrative says the classifier result remained violence.",
        ),
        incident(
            "inc-2",
            verdict="non-violence",
            confidence=0.72,
            packet_summary="Packet says no violent contact was present.",
            narrative="Stored narrative says the scene remained non-violence.",
        ),
    ]

    summary = summarize_incidents(records, "What happened last week?")

    assert summary.record_count == 2
    assert summary.matched_incidents == ["inc-1", "inc-2"]
    assert "inc-1" in summary.answer
    assert "0.9100" in summary.answer
    assert "stored packet summaries and stored narratives only" in summary.limitations_note


def test_summarize_incidents_handles_no_matches():
    summary = summarize_incidents([], "Any fights yesterday?")

    assert summary.record_count == 0
    assert summary.matched_incidents == []
    assert "No matching stored incidents" in summary.answer


def test_answer_historical_question_returns_ask_response_fields():
    result = answer_historical_question(
        "Tell me about violence today",
        records=[
            incident(
                "today-1",
                packet_summary="Two people pushing.",
                narrative="Stored narrative says pushing was described cautiously.",
                timestamp=NOW,
            )
        ],
        now=NOW,
    )

    assert set(result) == {
        "answer",
        "limitations_note",
        "matched_incidents",
        "record_count",
        "filters",
    }
    assert result["matched_incidents"] == ["today-1"]
    assert result["filters"]["verdict"] == "violence"


def test_format_explain_response_preserves_existing_fields():
    response = format_explain_response(
        {
            "narrative": "The candidate was accepted. The verdict was preserved.",
            "guardrail_status": "passed",
            "limitations_note": "No limitations.",
            "extra": "not part of demo field contract",
        }
    )

    assert response == {
        "narrative": "The candidate was accepted. The verdict was preserved.",
        "guardrail_status": "passed",
        "limitations_note": "No limitations.",
    }


def test_day4_historical_helpers_do_not_load_heavy_models():
    sys.modules.pop("torch", None)
    sys.modules.pop("transformers", None)

    answer_historical_question(
        "Tell me about incidents",
        records=[
            incident(
                "inc-1",
                packet_summary="Stored packet only.",
                narrative="Stored narrative only.",
            )
        ],
        now=NOW,
    )

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
