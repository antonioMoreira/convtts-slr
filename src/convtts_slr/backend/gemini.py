from ._llm import PydanticAIBackend


class GeminiBackend(PydanticAIBackend):
    """Gemini (Google AI Studio API) through pydantic-ai structured output.

    Needs GOOGLE_API_KEY, or the legacy GEMINI_API_KEY, and the `gemini` extra."""

    provider = "gemini"
    prefix = "google"  # pydantic-ai's provider name for the Gemini API
    default_model = "gemini-2.5-flash"
    extra = "gemini"
