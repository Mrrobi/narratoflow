"""narrato — compress huge LLM input context into dense intermediate representations."""

from narrato.pipeline import Compressor, CompressionResult, Decoder
from narrato.schemas import NarrativeFacts, get_schema

__version__ = "0.1.2"

__all__ = [
    "Compressor",
    "CompressionResult",
    "Decoder",
    "NarrativeFacts",
    "get_schema",
    "__version__",
]
