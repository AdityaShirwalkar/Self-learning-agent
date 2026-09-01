"""Application configuration for local development and cloud deployment."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# Cloud variables take precedence; this only fills in local values.
load_dotenv(BASE_DIR / ".env")

LOCAL_DB_PATH = str(BASE_DIR / "memory_db")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")
COLLECTION_NAME = "self_learning_agent_memories"

if bool(QDRANT_URL) != bool(QDRANT_API_KEY):
    raise RuntimeError(
        "Set both QDRANT_URL and QDRANT_API_KEY, or leave both unset for local Chroma storage."
    )

if QDRANT_URL:
    VECTOR_STORE_CONFIG = {
        "provider": "qdrant",
        "config": {"collection_name": COLLECTION_NAME, "url": QDRANT_URL, "api_key": QDRANT_API_KEY},
    }
else:
    VECTOR_STORE_CONFIG = {
        "provider": "chroma",
        "config": {"collection_name": COLLECTION_NAME, "path": LOCAL_DB_PATH},
    }

MEM0_CONFIG = {
    "vector_store": VECTOR_STORE_CONFIG,
    "embedder": {"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
    "llm": {
        "provider": "groq",
        "config": {
            "model": "openai/gpt-oss-120b",
            "temperature": 0.2,
            "max_tokens": 1000,
            "api_key": os.getenv("GROQ_API_KEY"),
        },
    },
}

CHAT_MODEL = "openai/gpt-oss-120b"
