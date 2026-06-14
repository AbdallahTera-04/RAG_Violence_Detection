import importlib
import os

MODULES = {
    "pydantic": "pydantic",
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "PIL",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "chromadb": "chromadb",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "torch": "torch",
    "pytest": "pytest",
    "python-dotenv": "dotenv",
    "sqlalchemy": "sqlalchemy",
}

failed = False
for package, module_name in MODULES.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: OK ({version})", flush=True)
    except Exception as exc:
        failed = True
        print(f"{package}: FAILED ({type(exc).__name__}: {exc})", flush=True)

os._exit(1 if failed else 0)
