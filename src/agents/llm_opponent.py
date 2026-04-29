"""LLM-based opponent using BAML with robust error handling."""

import logging
import os

from baml_py import ClientRegistry

from baml_client import b as baml_client

# Configure BAML client at runtime to handle Docker networking
_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_client_registry = ClientRegistry()
_client_registry.add_llm_client(
    name="Model",
    provider="openai-generic",
    options={
        "base_url": _ollama_url,
        "model": "qwen3.5:4b",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
)
_client_registry.set_primary("Model")

# Create configured BAML client
b = baml_client.with_options(client_registry=_client_registry)

logger = logging.getLogger(__name__)
