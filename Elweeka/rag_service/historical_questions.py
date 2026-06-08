"""Historical-question helpers for Elweeka Day 4.

The functions here operate on already stored incident records. They do not
create incident memory, perform real SQLite/FAISS access, load models, or
summarize with an LLM. Store and vector ranking integrations are dependency
injection points for future work.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol

from rag_service.guardrails import WEAPON_TERMS
from rag_service.schemas import (
    HistoricalQueryFilters,
    HistoricalSearchResult,
    HistoricalSummary,
    IncidentRecord,
)


class IncidentStore(Protocol):
    """Future SQLite-backed incident store interface."""

    def search(self, filters: HistoricalQueryFilters) -> Sequence[Any]:
        """Return records after metadata filtering."""


VectorRanker = Callable[[str, Sequence[IncidentRecord], int], Sequence[Any]]


def parse_historical_filters(
    question: str,
    *,
    now: datetime | None = None,
) -> HistoricalQueryFilters:
    """Parse simple time, verdict, and weapon filters from a question."""

    question_norm = _normalize(question)
    now = now or datetime.now()

    verdict = None
    if re.search(r"\bnon\s*violence\b|\bnonviolent\b|\bnon violent\b", question_norm):
        verdict = "non-violence"
    elif re.search(r"\bviolence\b|\bviolent\b|\bfight\b|\battack\b", question_norm):
        verdict = "violence"

    weapons = sorted(
        weapon for weapon in WEAPON_TERMS if _contains_word(question_norm, weapon)
    )
    start_time, end_time = _parse_time_window(question_norm, now)

    return HistoricalQueryFilters(
        verdict=verdict,
        weapons=weapons,
        start_time=start_time,
        end_time=end_time,
    )


def search_incidents(
    question: str,
    *,
    records: Iterable[Any] | None = None,
    incident_store: IncidentStore | Callable[[HistoricalQueryFilters], Sequence[Any]] | None = None,
    vector_ranker: VectorRanker | None = None,
    now: datetime | None = None,
    limit: int = 5,
) -> HistoricalSearchResult:
    """Filter incidents first, then rank candidates for a historical question."""

    filters = parse_historical_filters(question, now=now)
    source_records = _load_source_records(
        records=records,
        incident_store=incident_store,
        filters=filters,
    )
    filtered_records = _filter_records(source_records, filters)
    ranked_records = _rank_records(
        question=question,
        records=filtered_records,
        vector_ranker=vector_ranker,
        limit=limit,
    )

    return HistoricalSearchResult(
        question=question.strip(),
        filters=filters,
        records=ranked_records,
        limitations_note=(
            "Historical search used stored incident records only. "
            "Metadata filters were applied before ranking; vector ranking is an injected hook."
        ),
    )


def summarize_incidents(
    records: Sequence[Any],
    question: str,
    *,
    max_records: int = 3,
) -> HistoricalSummary:
    """Summarize matched incidents using stored packet summaries and narratives."""

    incident_records = [_coerce_record(record) for record in records]
    selected_records = incident_records[:max_records]

    if not selected_records:
        return HistoricalSummary(
            question=question.strip(),
            answer="No matching stored incidents were found for this historical question.",
            limitations_note=(
                "The answer is limited to existing stored incidents; no retrieval, "
                "model inference, or new evidence analysis was performed."
            ),
            matched_incidents=[],
            record_count=0,
        )

    incident_sentences = [
        _incident_summary_sentence(record) for record in selected_records
    ]
    answer = (
        f"Found {len(incident_records)} stored incident(s) relevant to the question. "
        + " ".join(incident_sentences)
    )
    if len(incident_records) > len(selected_records):
        answer += f" {len(incident_records) - len(selected_records)} additional incident(s) were omitted from this concise summary."

    return HistoricalSummary(
        question=question.strip(),
        answer=answer,
        limitations_note=(
            "This historical summary uses stored packet summaries and stored narratives only. "
            "Details may be heuristic when the original stored incident was heuristic."
        ),
        matched_incidents=[record.incident_id for record in selected_records],
        record_count=len(incident_records),
    )


def answer_historical_question(
    question: str,
    *,
    records: Iterable[Any] | None = None,
    incident_store: IncidentStore | Callable[[HistoricalQueryFilters], Sequence[Any]] | None = None,
    vector_ranker: VectorRanker | None = None,
    now: datetime | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Return a demo-friendly `/ask` historical-question response."""

    search_result = search_incidents(
        question,
        records=records,
        incident_store=incident_store,
        vector_ranker=vector_ranker,
        now=now,
        limit=limit,
    )
    summary = summarize_incidents(search_result.records, question)
    return format_ask_response(summary=summary, search_result=search_result)


def format_ask_response(
    *,
    summary: HistoricalSummary,
    search_result: HistoricalSearchResult,
) -> dict[str, Any]:
    """Map historical RAG output to stable demo `/ask` response fields."""

    return {
        "answer": summary.answer,
        "limitations_note": summary.limitations_note,
        "matched_incidents": summary.matched_incidents,
        "record_count": summary.record_count,
        "filters": search_result.filters.model_dump(),
    }


def format_explain_response(explanation_result: dict[str, Any]) -> dict[str, Any]:
    """Map Elweeka explanation output to stable demo `/explain` response fields."""

    response = {
        "narrative": explanation_result["narrative"],
        "guardrail_status": explanation_result["guardrail_status"],
        "limitations_note": explanation_result["limitations_note"],
    }
    if "guardrail_result" in explanation_result:
        response["guardrail_result"] = explanation_result["guardrail_result"]
    return response


def _load_source_records(
    *,
    records: Iterable[Any] | None,
    incident_store: IncidentStore | Callable[[HistoricalQueryFilters], Sequence[Any]] | None,
    filters: HistoricalQueryFilters,
) -> list[IncidentRecord]:
    if incident_store is not None:
        if hasattr(incident_store, "search"):
            raw_records = incident_store.search(filters)
        else:
            raw_records = incident_store(filters)
    else:
        raw_records = records or []

    return [_coerce_record(record) for record in raw_records]


def _filter_records(
    records: Sequence[IncidentRecord],
    filters: HistoricalQueryFilters,
) -> list[IncidentRecord]:
    result = []
    for record in records:
        if filters.verdict and record.verdict != filters.verdict:
            continue
        if filters.weapons and not all(
            _record_mentions_weapon(record, weapon) for weapon in filters.weapons
        ):
            continue
        if filters.start_time or filters.end_time:
            if record.timestamp is None:
                continue
            if filters.start_time and record.timestamp < filters.start_time:
                continue
            if filters.end_time and record.timestamp > filters.end_time:
                continue
        result.append(record)
    return result


def _rank_records(
    *,
    question: str,
    records: Sequence[IncidentRecord],
    vector_ranker: VectorRanker | None,
    limit: int,
) -> list[IncidentRecord]:
    if limit < 1:
        return []

    if vector_ranker is not None:
        return [_coerce_record(record) for record in vector_ranker(question, records, limit)][:limit]

    question_terms = _query_terms(question)
    scored = [
        (_lexical_score(question_terms, _record_text(record)), index, record)
        for index, record in enumerate(records)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in scored[:limit]]


def _incident_summary_sentence(record: IncidentRecord) -> str:
    stored_text = _first_sentence(record.narrative) or _first_sentence(record.packet_summary)
    return (
        f"Incident {record.incident_id} was stored as {record.verdict} "
        f"with confidence {record.confidence:.4f}; {stored_text}"
    )


def _parse_time_window(
    question_norm: str,
    now: datetime,
) -> tuple[datetime | None, datetime | None]:
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if _contains_word(question_norm, "today"):
        return today_start, now
    if _contains_word(question_norm, "yesterday"):
        start = today_start - timedelta(days=1)
        return start, today_start
    if "week ago" in question_norm:
        start = today_start - timedelta(days=8)
        end = today_start - timedelta(days=6)
        return start, end
    if "last week" in question_norm or "past week" in question_norm or "last 7 days" in question_norm:
        return now - timedelta(days=7), now
    if "last month" in question_norm or "past month" in question_norm:
        return now - timedelta(days=30), now

    return None, None


def _record_mentions_weapon(record: IncidentRecord, weapon: str) -> bool:
    record_weapons = " ".join(record.weapons)
    return _contains_word(_normalize(record_weapons), weapon) or _contains_word(
        _normalize(_record_text(record)), weapon
    )


def _record_text(record: IncidentRecord) -> str:
    references = " ".join(
        " ".join(
            str(part)
            for part in (
                reference.reference_id,
                reference.source,
                reference.title,
                reference.snippet,
            )
            if part
        )
        for reference in record.retrieved_references
    )
    return " ".join(
        [
            record.incident_id,
            record.verdict,
            record.packet_summary,
            record.narrative,
            " ".join(record.weapons),
            references,
            " ".join(str(value) for value in record.metadata.values()),
        ]
    )


def _coerce_record(record: Any) -> IncidentRecord:
    if isinstance(record, IncidentRecord):
        return record
    return IncidentRecord.model_validate(record)


def _query_terms(value: str) -> set[str]:
    stop_words = {
        "a",
        "about",
        "an",
        "and",
        "from",
        "in",
        "me",
        "of",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(value))
        if token not in stop_words and len(token) > 1
    }


def _lexical_score(query_terms: set[str], text: str) -> int:
    text_terms = _query_terms(text)
    return len(query_terms & text_terms)


def _first_sentence(value: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)[0].strip()
    return sentence


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


__all__ = [
    "answer_historical_question",
    "format_ask_response",
    "format_explain_response",
    "parse_historical_filters",
    "search_incidents",
    "summarize_incidents",
]
