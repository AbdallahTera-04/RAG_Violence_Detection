# Guardian Eye RAG - Elweeka Day 4 Implementation Notes

## Day 4 Objectives Completed

The project guideline assigns Day 4 to historical questions:

- connect RAG outputs to existing `/explain` and `/ask` response fields,
- implement historical incident search with simple time, verdict, and weapon filters,
- filter metadata first, then rank candidates,
- summarize incidents using stored packet summaries and stored narratives only,
- include limitations when details are heuristic.

This implementation keeps Day 4 inside Elweeka's explanation and guardrail boundary. It does not build a real incident store, retrieval pipeline, model loader, or demo backend.

## Files Modified

- `rag_service/schemas.py`
  - Added Pydantic contracts for stored historical incidents, parsed historical filters, search results, and historical summaries.
- `rag_service/historical_questions.py`
  - Added Day 4 historical question helpers and demo response adapters.
- `tests/test_historical_questions.py`
  - Added tests for filter parsing, filtered search, injected store/ranker hooks, grounded summaries, response fields, and no heavy model loading.
- `Day4_Implementation_Notes.md`
  - Added this implementation note.

## Architectural Decisions

Historical search is implemented with dependency injection:

- `incident_store` is the future SQLite hook.
- `vector_ranker` is the future FAISS/vector-search hook.
- If no vector ranker is provided, a deterministic lexical ranker is used for tests and local fallback.

The search flow follows the guideline:

1. Parse simple filters from the question.
2. Load records from an injected store or supplied records.
3. Apply metadata filters first.
4. Rank the filtered candidates.
5. Return a structured search result.

The summary flow is deliberately conservative:

- it uses stored `packet_summary` and `narrative` only,
- it does not inspect frames,
- it does not call a VLM or LLM,
- it includes a limitations note.

## Response Field Compatibility

`format_explain_response()` maps Elweeka explanation output to stable `/explain` fields:

- `narrative`
- `guardrail_status`
- `limitations_note`
- optional `guardrail_result`

`answer_historical_question()` and `format_ask_response()` produce stable `/ask` fields:

- `answer`
- `limitations_note`
- `matched_incidents`
- `record_count`
- `filters`

## Tests Added

Day 4 tests cover:

- verdict, weapon, and time filter parsing,
- filtering before vector ranking,
- injected store compatibility,
- summaries from stored packet/narrative only,
- no-match historical answers,
- `/ask` response field shape,
- `/explain` response field shape,
- no `torch` or `transformers` loading.

All Day 1-3 tests remain preserved.

## Limitations

- The lexical ranker is only a deterministic fallback, not a semantic FAISS search.
- The incident store is an injected interface, not a real SQLite implementation.
- Time parsing covers simple terms such as today, yesterday, last week, past week, last 7 days, week ago, and last month.
- Historical summaries are concise and grounded, not LLM-generated.
- Frame references are not re-opened or visually inspected.

## Future Integration Notes

Future work can connect:

- SQLite-backed `incident_store.search(filters)`,
- FAISS-backed `vector_ranker(question, records, limit)`,
- optional LLM summarization after VLM unload,
- future narrator models:
  - `Qwen2.5-VL-3B-Instruct-AWQ`,
  - `Qwen2.5-7B-Instruct`,
  - `Llama-3.1-8B-Instruct`.

Any future LLM summarizer must preserve the same rule: use stored packet summaries and stored narratives only, and never modify classifier verdicts or confidence values.

## Intentionally Not Implemented

- real SQLite incident memory,
- real FAISS retrieval,
- evidence packet generation,
- incident insertion,
- orchestration,
- historical LLM summarization,
- real VLM inference,
- real LLM inference,
- model loading,
- frontend/demo backend changes.

Elweeka Day 4 historical-question support is complete within the explanation-layer boundary.
