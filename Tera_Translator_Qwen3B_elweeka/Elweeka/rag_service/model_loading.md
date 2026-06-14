# Guardian Eye RAG Model-Loading Design

## Purpose

This document defines the model-loading strategy for the Guardian Eye explanation layer, owned by Elweeka.

The classifier is the authority. Its `verdict` and `confidence` are immutable facts. The VLM/LLM layer is only a narrator: it may explain the classifier result using grounded evidence, but it must never re-decide violence vs non-violence, change confidence, or override the classifier.

This is a design document only. It does not implement Day 2 functions, load models, or build `generate_clip_explanation()`.

## Chosen Components

- Primary VLM: `Qwen2.5-VL-3B-Instruct-AWQ`
- Text LLM options:
  - `Qwen2.5-7B-Instruct` 4-bit
  - `Llama-3.1-8B-Instruct` INT4
- Embedding model: `BAAI/bge-small-en-v1.5`
- Vector search: FAISS
- Metadata store: SQLite

## Required Execution Flow

```text
Classifier
v
Unload classifier
v
Reference retrieval
v
VLM explanation
v
Unload VLM
v
LLM historical summarization if needed
```

The main safety property is that classifier, VLM, and LLM weights must not be resident in VRAM at the same time.

## Runtime Strategy

1. Run the violence classifier first.
2. Persist the classifier output as fixed fields:
   - `verdict`
   - `confidence`
   - classifier-side evidence signals used to build `packet_summary`
3. Fully unload the classifier before loading any large narrator model.
4. Run reference retrieval using the embedding model, FAISS, and SQLite metadata.
5. Load the VLM only when frame-grounded explanation is needed.
6. Generate a 2-4 sentence grounded explanation using:
   - fixed classifier verdict
   - fixed classifier confidence
   - evidence packet summary
   - retrieved reference snippets
   - up to 16 stored frames
7. Run guardrails against the generated narrative.
8. Unload the VLM immediately after the explanation attempt.
9. Load a text LLM only if historical incident summarization is required.
10. Unload the text LLM after the summarization step.

Text-only fallback must skip VLM loading when frames are unavailable or when VRAM is insufficient. In that mode, the narrator uses only `verdict`, `confidence`, `packet_summary`, and `retrieved_references`.

## VRAM Safety Rules

- Never keep classifier, VLM, and LLM in VRAM at the same time.
- Never load the VLM before classifier teardown is complete.
- Never load the text LLM while the VLM is still resident.
- Prefer quantized models for narrator tasks:
  - AWQ for the VLM
  - 4-bit or INT4 for text LLM
- Load heavy models sequentially and on demand.
- Release model objects, tokenizer/processor objects, and cached tensors during unload.
- After unloading a heavy model, explicitly clear CUDA cache when running on GPU.
- Historical summarization must run after VLM unload, not in parallel with VLM explanation.
- If GPU memory is insufficient, fall back to text-only explanation instead of risking an unstable runtime.
- Do not use the narrator model to revise classifier verdict or confidence under any error condition.

## Sequential Loading Lifecycle

### 1. Classifier Stage

Input:

- video clip or preprocessed classifier input

Output:

- immutable `verdict`
- immutable `confidence`
- classifier evidence for `packet_summary`
- optional stored frame references

Lifecycle requirement:

- After classifier output is persisted, unload classifier weights before retrieval and narration.

### 2. Retrieval Stage

Input:

- `packet_summary`
- incident metadata
- optional query text derived from classifier-side evidence

Models/resources:

- `BAAI/bge-small-en-v1.5`
- FAISS index
- SQLite metadata store

Output:

- `retrieved_references`

Lifecycle requirement:

- Embedding model should be lightweight relative to narrator models, but still should be loaded on demand if memory pressure exists.
- FAISS and SQLite stay CPU/local-first.

### 3. VLM Explanation Stage

Input:

- immutable `verdict`
- immutable `confidence`
- `packet_summary`
- `retrieved_references`
- `frames_ref`, up to 16 stored frames

Model:

- `Qwen2.5-VL-3B-Instruct-AWQ`

Output:

- candidate 2-4 sentence explanation narrative

Lifecycle requirement:

- Load VLM only after classifier unload.
- Unload VLM immediately after candidate explanation and guardrail validation/regeneration attempts.
- If frames are empty, skip VLM and use text-only fallback design.

### 4. Optional LLM Historical Summary Stage

Input:

- historical incident records
- retrieved reference summaries
- current fixed classifier output if needed for context

Model options:

- `Qwen2.5-7B-Instruct` 4-bit
- `Llama-3.1-8B-Instruct` INT4

Output:

- historical incident summary

Lifecycle requirement:

- Load text LLM only after VLM unload.
- Unload text LLM immediately after summary generation.

## Function Design Notes

These are Day 2 implementation targets only.

### `load_vlm()`

Expected input:

- `model_id`: default `Qwen2.5-VL-3B-Instruct-AWQ`
- `device`: target device, usually `cuda` if available
- `quantization`: AWQ-compatible loading configuration
- optional local model path for offline deployments

Expected output:

- loaded VLM model handle
- processor/tokenizer handle
- metadata describing device, dtype, quantization, and source path

Design notes:

- Must assert classifier is not loaded in VRAM.
- Must avoid loading if frames are unavailable and text-only fallback is selected.
- Must support local/offline model paths.
- Must fail closed: if loading fails, return an error that triggers text-only fallback, not verdict reinterpretation.

### `unload_vlm()`

Expected input:

- VLM model handle
- processor/tokenizer handle
- optional device/cache manager

Expected output:

- unload result with success flag and diagnostic message

Design notes:

- Delete model and processor references.
- Clear GPU cache when CUDA is active.
- Make repeated calls safe.
- Must not alter `verdict`, `confidence`, candidate narrative, or guardrail result.

### `load_llm()`

Expected input:

- `model_id`: one of:
  - `Qwen2.5-7B-Instruct` 4-bit
  - `Llama-3.1-8B-Instruct` INT4
- `device`: target device
- quantization configuration
- optional local model path for offline deployments

Expected output:

- loaded LLM model handle
- tokenizer handle
- metadata describing device, dtype, quantization, and source path

Design notes:

- Must assert VLM is not loaded in VRAM.
- Must only load when historical summarization is needed.
- Must support local/offline model paths.
- Must not be used to re-score the incident or rewrite classifier fields.

### `unload_llm()`

Expected input:

- LLM model handle
- tokenizer handle
- optional device/cache manager

Expected output:

- unload result with success flag and diagnostic message

Design notes:

- Delete model and tokenizer references.
- Clear GPU cache when CUDA is active.
- Make repeated calls safe.
- Must not alter classifier outputs or stored explanation outputs.

## Loader Inputs And Outputs

| Function | Inputs | Outputs |
| --- | --- | --- |
| `load_vlm()` | model id/path, device, quantization config, offline flag | VLM handle, processor/tokenizer handle, load metadata |
| `unload_vlm()` | VLM handle, processor/tokenizer handle, cache/device manager | unload status, diagnostics |
| `load_llm()` | model id/path, device, quantization config, offline flag | LLM handle, tokenizer handle, load metadata |
| `unload_llm()` | LLM handle, tokenizer handle, cache/device manager | unload status, diagnostics |

No loader should accept or return a modified classifier verdict or confidence.

## Error-Handling Design

### Model Load Failure

Required behavior:

- Record the error.
- Do not retry in a loop that risks VRAM fragmentation.
- Do not modify verdict or confidence.
- For VLM failure, fall back to text-only explanation if `packet_summary` and `retrieved_references` are available.
- For LLM historical summary failure, omit the historical summary and return a clear limitation note.

### Out-Of-Memory Failure

Required behavior:

- Stop the current model-loading attempt.
- Unload any partially loaded model objects.
- Clear CUDA cache if applicable.
- Fall back to the lighter path:
  - VLM OOM: text-only explanation
  - LLM OOM: skip historical summary
- Preserve classifier verdict and confidence exactly.

### Missing Frames

Required behavior:

- Do not load VLM.
- Use text-only fallback.
- Include a limitation note that frame evidence was unavailable.

### Retrieval Failure

Required behavior:

- Continue with `packet_summary` only when safe.
- Include a limitation note that retrieved references were unavailable.
- Do not infer unsupported details to compensate for missing retrieval.

### Guardrail Failure

Required behavior:

- Reject the candidate explanation.
- Request regeneration using only:
  - fixed `verdict`
  - fixed `confidence`
  - `packet_summary`
  - `retrieved_references`
  - `frames_ref`
- Do not change verdict or confidence.
- Do not generate a replacement inside the guardrail layer.

## Offline-Friendly Requirements

- Prefer local model paths and local cache directories.
- Keep FAISS and SQLite local.
- Do not require network access during normal execution.
- Store model identifiers and local paths in configuration, not hardcoded runtime logic.
- Make failure modes explicit when a model is not available locally.

## Day 2 TODOs

- TODO: Implement a lightweight model registry/config object for local paths and model IDs.
- TODO: Implement `load_vlm()` with AWQ-compatible loading and offline path support.
- TODO: Implement `unload_vlm()` with safe repeated unload and CUDA cache cleanup.
- TODO: Implement `load_llm()` for one selected text LLM option.
- TODO: Implement `unload_llm()` with safe repeated unload and CUDA cache cleanup.
- TODO: Add runtime state checks that prevent classifier/VLM/LLM overlap in VRAM.
- TODO: Add text-only fallback path when frames are empty or VLM loading fails.
- TODO: Add structured loader diagnostics for setup reports and debugging.
- TODO: Add small integration tests with mocked model handles only.
