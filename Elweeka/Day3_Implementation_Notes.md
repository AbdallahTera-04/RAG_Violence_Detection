# Guardian Eye RAG - Elweeka Day 3 Implementation Notes

## Scope

Day 3 strengthens the Elweeka guardrail and regeneration path while staying inside the explanation layer. No retrieval, evidence packet generation, orchestration, model loading, real VLM inference, or real LLM inference was added.

## What Was Strengthened

- Guardrails now treat retrieved reference snippets as valid grounding support for objects, actions, people claims, exact timing, and responsibility claims.
- People-count validation now rejects narratives that claim more people than the evidence supports.
- Action detection was expanded with common variants such as hit, slap, and shove.
- Responsibility/blame detection was expanded with blame/fault language.
- Regeneration metadata is now exposed through the validated output structure when a candidate explanation is rejected.

## Guardrail Categories Covered

The guardrail layer checks for:

- verdict contradiction,
- confidence modification,
- invented weapons,
- invented objects,
- invented people and unsupported people counts,
- unsupported actions,
- unsupported exact timing,
- unsupported responsibility or blame claims.

Violations return `regenerate_required` and include clear rejection reasons plus a regeneration instruction.

## Regeneration Behavior

Regeneration is only an instruction/hook. The guardrail layer does not rewrite the explanation and does not call a model.

When `generate_clip_explanation()` receives a rejected candidate:

- the unsafe candidate is not accepted as final output,
- `verdict` and `confidence` are preserved exactly,
- `guardrail_status` is set to `regenerate_required`,
- structured guardrail issues are included,
- the regeneration instruction is included for a future narrator client.

No retry loop is implemented.

## Text-Only Fallback

Text-only fallback remains active when `frames_ref` is missing or empty. In that mode, Elweeka does not attempt a VLM path and uses only:

- fixed classifier verdict,
- fixed classifier confidence,
- packet summary,
- retrieved references.

The output limitation note states that text-only fallback was used.

## Future VLM Integration Points

The `narrator_client` injection point remains the future integration boundary for:

- `Qwen2.5-VL-7B-Instruct-AWQ`,
- `Qwen2.5-7B-Instruct`,
- `Llama-3.1-8B-Instruct`.

A future client can consume the formatted prompt messages and, after a `regenerate_required` result, use the regeneration instruction in a controlled retry policy. That retry policy is intentionally outside Day 3.

## Limitations

- Guardrails are deterministic and pattern-based.
- Frame references are treated as identifiers, not parsed visual evidence.
- The fallback narrative is deterministic and not a real visual explanation.
- Retrieved references are assumed to be provided by another module.
- The evidence packet is assumed to be deterministic and already built.

## Intentionally Not Implemented

- retrieval,
- evidence packet generation,
- incident memory,
- orchestration,
- historical summarization,
- real VLM inference,
- real LLM inference,
- model loading,
- frontend/demo code.

Elweeka Day 3 keeps the explanation layer stricter while preserving future model integration boundaries.
