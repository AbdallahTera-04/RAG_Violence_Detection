# Qwen VLM Model Change

## Output Location

This separate variant was created at:

`D:\RAG_Graduation_Translator\RAG_Violence_Detection\Tera_Translator_Qwen3B`

The original `Tera_Translator`, `Elweeka`, and `tera_notebooks` folders were
not modified.

## Before

- VLM model: `Qwen2.5-VL-7B-Instruct-AWQ`
- VLM local path: `models/qwen2.5-vl-7b-instruct-awq`

## After

- VLM model: `Qwen2.5-VL-3B-Instruct-AWQ`
- VLM local path: `models/qwen2.5-vl-3b-instruct-awq`

## Scope

Only references to the VLM model and its corresponding lowercase local path
were changed in the copied `Elweeka` documents and `tera_notebooks` setup
notebook.

The separate text model `Qwen2.5-7B-Instruct` was intentionally left
unchanged. No Python runtime logic, translator behavior, RAG behavior,
dependencies, tests, data, or indexes were changed.
