"""Re-ranking strategies for retrieved candidate evidence."""

from artificial_memory_gen.compiler.rerank.cross_encoder import (
    DEFAULT_MODEL,
    CrossEncoderReranker,
    CrossEncoderUnavailable,
    passage_text,
)
from artificial_memory_gen.compiler.rerank.ollama_listwise import (
    OllamaListwiseReranker,
    model_available,
)

__all__ = [
    "DEFAULT_MODEL",
    "CrossEncoderReranker",
    "CrossEncoderUnavailable",
    "OllamaListwiseReranker",
    "model_available",
    "passage_text",
]
