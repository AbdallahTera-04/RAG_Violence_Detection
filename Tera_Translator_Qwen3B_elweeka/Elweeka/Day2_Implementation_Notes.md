# Guardian Eye RAG - Elweeka Day 2 Implementation Notes

## Scope

Day 2 implements E1: a production-clean explanation service boundary for Elweeka. It connects Day 1 schemas, prompt formatting, and guardrails without loading real models or implementing retrieval/orchestration.

## Implementation Choices

- `rag_service/explanation_service.py` contains `generate_clip_explanation()`.
- Inputs are validated with `ExplanationInput` from `rag_service/schemas.py`.
- Narrator messages are formatted with `format_narrator_prompt()` from `rag_service/prompts.py`.
- Candidate narratives are validated with `validate_narrative()` from `rag_service/guardrails.py`.
- Outputs are validated with `ExplanationOutput` from `rag_service/schemas.py`.
- The public return value is a plain dictionary produced from the validated Pydantic output.

## How `generate_clip_explanation()` Works

1. Accept fixed classifier output:
   - `verdict`
   - `confidence`
2. Validate `packet_summary`, `retrieved_references`, and `frames_ref`.
3. Detect text-only fallback mode when `frames_ref` is missing or empty.
4. Format narrator prompts using the Day 1 prompt templates.
5. Use an injected `narrator_client` if provided.
6. If no client is provided, produce a deterministic mock fallback narrative.
7. Run guardrails against the candidate narrative.
8. Return a validated output containing:
   - `narrative`
   - `guardrail_status`
   - `limitations_note`

## Text-Only Fallback

If `frames_ref` is `None` or an empty list, the service does not attempt VLM behavior. It formats the prompt with text-only context and uses only:

- fixed `verdict`
- fixed `confidence`
- `packet_summary`
- `retrieved_references`

The output limitation note explicitly states that text-only fallback was used.

## Guardrail Enforcement

Guardrails are always applied after candidate narration.

If guardrails pass:

- return the candidate explanation,
- set `guardrail_status` to `passed`,
- preserve verdict and confidence unchanged.

If guardrails return `regenerate_required`:

- do not modify verdict or confidence,
- do not accept the unsafe candidate as the final narrative,
- return a safe guarded rejection narrative,
- include guardrail reasons in `limitations_note`,
- include structured guardrail issues in the optional output metadata.

## Future VLM Hook Points

The `narrator_client` parameter is the integration point for future model clients. A future client can expose either:

- `generate(messages=..., context=..., text_only=...)`, or
- a callable interface returning a narrative string or a dictionary with `narrative`.

This is compatible with future narrator backends such as:

- `Qwen2.5-VL-3B-Instruct-AWQ`
- `Qwen2.5-7B-Instruct`
- `Llama-3.1-8B-Instruct`

## Known Limitations

- The deterministic mock fallback is not a real visual explanation.
- Guardrails are deterministic and pattern-based, not semantic proof.
- The service assumes references were already retrieved.
- The service assumes the evidence packet was already built.
- Historical summarization is not connected in E1.

## Intentionally Not Implemented

- retrieval,
- evidence packet generation,
- incident memory,
- orchestration,
- historical summarization,
- real VLM/LLM inference,
- model loading,
- frontend/demo logic.

Day 2 E1 keeps Elweeka focused on explanation formatting, fallback behavior, guardrail enforcement, and future narrator-client compatibility.
