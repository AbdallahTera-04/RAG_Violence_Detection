"""Pydantic contracts for the Elweeka explanation layer.

These schemas define data exchanged with upstream classifier/evidence systems
and downstream consumers. They intentionally contain no orchestration,
retrieval, evidence-packet building, or model-generation logic.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Verdict = Literal["violence", "non-violence"]
GuardrailStatus = Literal["passed", "failed", "regenerate_required"]


class RetrievedReference(BaseModel):
    """A grounded reference retrieved for explanation context."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    title: str | None = Field(default=None, min_length=1)
    snippet: str | None = Field(default=None, min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailIssue(BaseModel):
    """A guardrail finding from validating the generated explanation."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    severity: Literal["info", "warning", "error"] = "error"


class GuardrailResult(BaseModel):
    """Guardrail outcome attached to an explanation response."""

    model_config = ConfigDict(extra="forbid")

    status: GuardrailStatus
    issues: list[GuardrailIssue] = Field(default_factory=list)
    regeneration_instruction: str | None = None


class ExplanationInput(BaseModel):
    """Input contract for Elweeka narration.

    The classifier verdict and confidence are final. Elweeka may explain them
    but must not reinterpret or replace them.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    packet_summary: str = Field(..., min_length=1)
    retrieved_references: list[RetrievedReference] = Field(default_factory=list)
    frames_ref: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Stored frame identifiers or paths. Empty list is valid for "
            "text-only fallback; up to 16 frames are supported."
        ),
    )

    @field_validator("packet_summary")
    @classmethod
    def packet_summary_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("packet_summary must be non-empty")
        return value

    @field_validator("frames_ref")
    @classmethod
    def frame_refs_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not frame_ref.strip() for frame_ref in value):
            raise ValueError("frames_ref entries must be non-empty strings")
        return value


class ExplanationOutput(BaseModel):
    """Output contract for Elweeka's grounded explanation."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(..., min_length=1)
    guardrail_status: GuardrailStatus
    limitations_note: str = Field(..., min_length=1)
    guardrail_result: GuardrailResult | None = None

    @field_validator("narrative")
    @classmethod
    def narrative_should_be_two_to_four_sentences(cls, value: str) -> str:
        value = value.strip()
        sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", value)
            if sentence.strip()
        ]
        if not 2 <= len(sentences) <= 4:
            raise ValueError("narrative must contain 2 to 4 sentences")
        return value

    @field_validator("limitations_note")
    @classmethod
    def limitations_note_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("limitations_note must be non-empty")
        return value


class IncidentRecord(BaseModel):
    """Stored incident data used for historical Elweeka questions."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    incident_id: str = Field(..., min_length=1)
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    packet_summary: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)
    timestamp: datetime | None = None
    retrieved_references: list[RetrievedReference] = Field(default_factory=list)
    weapons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("packet_summary", "narrative")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("incident text fields must be non-empty")
        return value


class HistoricalQueryFilters(BaseModel):
    """Simple parsed filters for historical incident search."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict | None = None
    weapons: list[str] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None


class HistoricalSearchResult(BaseModel):
    """Filtered and ranked incident search output."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    filters: HistoricalQueryFilters
    records: list[IncidentRecord] = Field(default_factory=list)
    limitations_note: str = Field(..., min_length=1)


class HistoricalSummary(BaseModel):
    """Grounded answer for historical questions."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    limitations_note: str = Field(..., min_length=1)
    matched_incidents: list[str] = Field(default_factory=list)
    record_count: int = Field(..., ge=0)
