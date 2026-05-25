"""narrato — compress huge LLM input context into dense intermediate representations."""

from narrato.pipeline import CompressionResult, Compressor, Decoder
from narrato.profiles import Profile, get_profile, list_profiles, register_profile
from narrato.schemas import NarrativeFacts, get_schema, list_presets

__version__ = "0.3.0"

__all__ = [
    "CompressionResult",
    "Compressor",
    "Decoder",
    "NarrativeFacts",
    "Profile",
    "__version__",
    "get_profile",
    "get_schema",
    "list_presets",
    "list_profiles",
    "register_profile",
]
