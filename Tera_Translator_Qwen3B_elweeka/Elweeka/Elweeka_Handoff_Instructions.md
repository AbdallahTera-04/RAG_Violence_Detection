# Guardian Eye RAG - Elweeka Handoff Instructions

## Scope

This handoff covers Elweeka only: explanation generation boundaries, guardrail validation, text-only fallback, historical question helpers, and demo-compatible response shapes.

Team-wide integration is intentionally not performed in this project folder yet.

## Replace Mock `/explain` Behavior With

Use:

```python
from explanation_service import generate_clip_explanation
```

Call:

```python
generate_clip_explanation(
    verdict=classifier_verdict,
    confidence=classifier_confidence,
    packet_summary=evidence_packet_summary,
    retrieved_references=reference_snippets,
    frames_ref=stored_frame_refs,
    narrator_client=future_vlm_or_llm_client,
)
```

Expected output fields:

- `narrative`
- `guardrail_status`
- `limitations_note`
- optional `guardrail_result`

Mock/demo function to replace:

- Replace the existing mock explanation response producer behind `/explain` with `generate_clip_explanation(...)`.
- Keep the existing UI and response field style.
- Do not let the narrator client alter `verdict` or `confidence`.

Future model hook:

- Pass a `narrator_client` that implements `generate(messages=..., context=..., text_only=...)`.
- The future client may wrap `Qwen2.5-VL-3B-Instruct-AWQ`, `Qwen2.5-7B-Instruct`, or `Llama-3.1-8B-Instruct`.
- Model loading must remain sequential and on demand.

## Replace Mock `/ask` Behavior With

Use:

```python
from rag_service.historical_questions import answer_historical_question
```

Call:

```python
answer_historical_question(
    question=user_question,
    incident_store=future_sqlite_store,
    vector_ranker=future_faiss_ranker,
)
```

Expected output fields:

- `answer`
- `limitations_note`
- `matched_incidents`
- `record_count`
- `filters`

Mock/demo function to replace:

- Replace the existing mock historical ask response producer behind `/ask` with `answer_historical_question(...)`.
- Keep the existing UI and response field style.
- Use stored packet summaries and stored narratives only.

Future retrieval hooks:

- `incident_store.search(filters)` should perform the SQLite metadata-filter stage.
- `vector_ranker(question, records, limit)` should perform the FAISS/vector-ranking stage after metadata filtering.

## Guardrail Contract

If guardrails pass:

- `guardrail_status = "passed"`
- return the candidate narrative.

If guardrails reject the candidate:

- `guardrail_status = "regenerate_required"`
- return a safe guarded rejection narrative,
- include structured issues and a regeneration instruction,
- do not execute regeneration inside the guardrail layer.

The classifier `verdict` and `confidence` are immutable in every case.

## Files To Keep Stable For Integration

- `explanation_service.py`
- `rag_service/explanation_service.py`
- `rag_service/prompts.py`
- `rag_service/guardrails.py`
- `rag_service/schemas.py`
- `rag_service/historical_questions.py`

## Final Integration Boundaries

Do not replace these with direct model calls. Keep them as the Elweeka boundary:

- `/explain` -> `generate_clip_explanation(...)`
- `/ask` -> `answer_historical_question(...)`

The demo frontend should not be rebuilt for Elweeka integration.
