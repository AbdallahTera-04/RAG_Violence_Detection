# Guardian Eye RAG - Elweeka Day 1 Design

## Scope

Elweeka is the explanation and guardrail layer for Guardian Eye. The classifier is the authority: its `verdict` and `confidence` are fixed facts. The VLM/LLM layer may narrate and summarize, but it must never re-decide violence vs non-violence or modify classifier confidence.

This document is Day 1 design only. It does not implement `generate_clip_explanation()`, VLM inference, orchestration, or model-loading code.

## 1. Model-Loading Strategy

Execution must be sequential and on demand:

```text
Classifier
v
Unload classifier
v
Retrieve references
v
Load VLM on demand
v
Generate grounded explanation candidate
v
Unload VLM
v
Load text LLM only if historical summarization is needed
v
Unload text LLM
```

Rules:

- Run the classifier first and persist immutable `verdict` and `confidence`.
- Unload classifier weights before retrieval or narrator model loading.
- Retrieve references with local FAISS and SQLite metadata before explanation.
- Load `Qwen2.5-VL-7B-Instruct-AWQ` only when frame-grounded explanation is needed.
- Unload the VLM immediately after explanation and guardrail validation/regeneration attempts.
- Load a text LLM only for historical incident summarization.
- Never keep classifier, VLM, and text LLM resident in VRAM at the same time.

## 2. Prompt Strategy

The narrator prompt defines Elweeka as an explanation-only layer.

Prompt rules:

- Classifier `verdict` is immutable.
- Classifier `confidence` is immutable.
- VLM/LLM is narrator only, not a second classifier.
- Explanation may use only:
  - the 16 stored frames,
  - deterministic evidence packet summary,
  - retrieved reference snippets.
- The model must not invent people, weapons, objects, actions, responsibility, or exact timing.
- Output must be one paragraph of 2-4 grounded sentences.
- If frame evidence, retrieval, or visual detail is incomplete, the explanation must mention limitations.

## 3. Guardrail Strategy

The guardrail layer validates candidate narratives after generation. It does not generate replacements and does not change classifier outputs.

Guardrails detect:

- verdict contradiction,
- confidence modification,
- invented people,
- invented weapons,
- invented objects,
- unsupported actions,
- unsupported exact timing,
- unsupported responsibility claims.

Behavior:

- If safe, return `passed`.
- If unsafe, return `regenerate_required` with clear rejection reasons.
- If unsafe, provide a regeneration instruction that tells the narrator to rewrite using only `verdict`, `confidence`, `packet_summary`, `retrieved_references`, and `frames_ref`.
- Verdict and confidence remain unchanged in all outcomes.

## 4. JSON Contracts

Implemented in `rag_service/schemas.py`.

### `RetrievedReference`

Represents a retrieved grounding snippet.

Key fields:

- `reference_id`
- `source`
- `title`
- `snippet`
- `score`
- `metadata`

### `ExplanationInput`

Input to the Elweeka explanation layer.

Key fields:

- `verdict`: `"violence"` or `"non-violence"`
- `confidence`: float from `0.0` to `1.0`
- `packet_summary`: non-empty deterministic evidence summary
- `retrieved_references`: list of `RetrievedReference`
- `frames_ref`: up to 16 stored frame references, empty allowed for text-only fallback

### `ExplanationOutput`

Output from the explanation layer.

Key fields:

- `narrative`: 2-4 grounded sentences
- `guardrail_status`: `"passed"`, `"failed"`, or `"regenerate_required"`
- `limitations_note`
- optional `guardrail_result`

### `GuardrailResult`

Represents validation outcome.

Key fields:

- `status`
- `reasons` or `issues`
- optional `regeneration_instruction`

## 5. Assumptions

- The 16 stored frames are already available before Elweeka runs.
- The evidence packet is deterministic and produced before prompt formatting.
- Reference snippets are retrieved before explanation.
- The classifier output is already produced by another module.
- Elweeka receives classifier output as fixed input, not as something to verify or revise.
- Local/offline execution is preferred.

## 6. Future Day 2 Plan

Day 2 implementation should:

- implement `generate_clip_explanation()`,
- connect schemas, prompt formatting, model loaders, and guardrails,
- add text-only fallback when frames are empty or VLM loading fails,
- add mocked tests for model-loading lifecycle,
- add validation tests for prompt and guardrail behavior,
- add integration tests without loading real heavy models.

Elweeka Day 1 design is complete.
