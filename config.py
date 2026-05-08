"""
config.py — Shared configuration helpers for Day 22 Lab
=========================================================
Loads environment variables from .env and exposes typed config objects
used by all four lab scripts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env (searches current dir, then one level up) ─────────────────────
_here = Path(__file__).resolve().parent
for _candidate in [_here / ".env", _here.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=True)
        break

# ── Set LangSmith tracing vars BEFORE any LangChain imports ─────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGSMITH_PROJECT", "day22-langsmith-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


class Config:
    """Central configuration object consumed by all lab scripts."""

    # OpenRouter / OpenAI-compatible LLM endpoint
    openai_api_key:   str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url:  str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    openai_model:     str = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")

    # HuggingFace local embedding model (free, no API key needed)
    embedding_model:  str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LangSmith
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "day22-langsmith-lab")

    # Paths
    base_dir:   Path = _here
    data_dir:   Path = _here / "data"
    evidence_dir: Path = _here / "evidence"


cfg = Config()


def validate_config(require_langsmith: bool = True) -> None:
    """Raise SystemExit if required env vars are missing."""
    errors = []
    if not cfg.openai_api_key or cfg.openai_api_key == "your-openrouter-api-key-here":
        errors.append("OPENAI_API_KEY is not set in .env")
    if require_langsmith and (
        not cfg.langsmith_api_key or cfg.langsmith_api_key == "your-langsmith-api-key-here"
    ):
        errors.append("LANGSMITH_API_KEY is not set in .env")
    if errors:
        for e in errors:
            print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    validate_config()
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {cfg.langsmith_project}")
    print(f"   OpenAI endpoint   : {cfg.openai_base_url}")
    print(f"   Default LLM model : {cfg.openai_model}")
    print(f"   Embedding model   : {cfg.embedding_model}")
