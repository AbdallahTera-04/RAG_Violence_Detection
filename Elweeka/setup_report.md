# Guardian Eye RAG Dependency Setup Report

Date: 2026-06-08

## Environment

- Project folder: `D:\sara\graduation\RAG`
- Target environment: project virtual environment at `.venv`
- Python version: `Python 3.11.9`
- Pip version before install: `pip 24.0`

## Files Reviewed First

- `C:\Users\farah\Downloads\dependency_status.txt`
- `C:\Users\farah\Downloads\check_dependencies.py`
- `C:\Users\farah\Downloads\requirements-rag.txt`

The provided `dependency_status.txt` showed all RAG dependencies as already available in a previous environment. The live system `python` checked from this project folder was different and initially only had `pip==24.0` and `setuptools==65.5.0`, so the RAG packages were missing there.

## Packages Already Existing Before Install

In the newly created project `.venv`, these packages already existed before installing RAG requirements:

- `pip==24.0`
- `setuptools==65.5.0`

No direct RAG dependency from `requirements-rag.txt` was already installed in the new `.venv`.

## Installed Direct RAG Requirements

Installed into `.venv` from `C:\Users\farah\Downloads\requirements-rag.txt`:

- `pydantic==2.13.4`
- `numpy==2.4.6`
- `pandas==3.0.3`
- `pillow==12.2.0`
- `sentence-transformers==5.5.1`
- `faiss-cpu==1.14.2`
- `chromadb==1.5.9`
- `transformers==5.10.2`
- `accelerate==1.13.0`
- `bitsandbytes==0.49.2`
- `torch==2.12.0`
- `pytest==9.0.3`
- `python-dotenv==1.2.2`
- `sqlalchemy==2.0.50`

Transitive dependencies were installed by pip as required by the direct RAG packages. No frontend/demo dependencies such as React, Vite, Tailwind, Axios, Recharts, FFmpeg overlay tooling, or OpenCV overlay work were installed.

## Failed Packages

None.

`pip check` result:

```text
No broken requirements found.
```

## Torchvision Note

`torchvision` was not installed. It is intentionally not pinned in `requirements-rag.txt`, and it is not needed for the RAG-only explanation layer focused on VLM/LLM loading, text-only fallback, narrator prompts, guardrails, grounded explanation output, and historical incident summaries.

This avoids disturbing the working environment with a package that can be sensitive to the exact `torch` and CUDA build. If a future GPU VLM path requires `torchvision`, install a version that exactly matches the target `torch` and CUDA runtime in that GPU environment.

## Final Verification

Command run:

```powershell
.\.venv\Scripts\python.exe C:\Users\farah\Downloads\check_dependencies.py
```

Result:

```text
pydantic: OK (2.13.4)
numpy: OK (2.4.6)
pandas: OK (3.0.3)
pillow: OK (12.2.0)
sentence-transformers: OK (5.5.1)
faiss-cpu: OK (1.14.2)
chromadb: OK (1.5.9)
transformers: OK (5.10.2)
accelerate: OK (1.13.0)
bitsandbytes: OK (0.49.2)
torch: OK (2.12.0+cpu)
pytest: OK (9.0.3)
python-dotenv: OK (unknown)
sqlalchemy: OK (2.0.50)
```

Final status: PASS.

Use this environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```
