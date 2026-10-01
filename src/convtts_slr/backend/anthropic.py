from ._llm import PydanticAIBackend


class AnthropicBackend(PydanticAIBackend):
    """Claude through pydantic-ai structured output. Needs ANTHROPIC_API_KEY.

    Also the natural *independent* checker for another System One (see verify.py)."""

    provider = "anthropic"
    prefix = "anthropic"
    default_model = "claude-sonnet-5"
    extra = "llm"
