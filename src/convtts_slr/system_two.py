"""System Two: generative extraction, used only where a bounded answer is impossible
(numbers, names, free text). Every value must come with a verbatim quote so that
verify.py can check it against the parsed paper."""

from collections.abc import Callable
from typing import Protocol

from .models import DatasetFacts, Paper

_INSTRUCTIONS = (
    "Extract facts about the speech dataset described in the paper text. For each field, set "
    "`value` and copy into `quote` ONE sentence from the text, verbatim, that supports it. "
    "If the text does not state a field, leave both value and quote null. Never infer numbers "
    "that are not written in the text."
)


class FactsExtractor(Protocol):
    name: str

    def extract(self, paper: Paper, text: str) -> DatasetFacts: ...


class LLMFactsExtractor:
    name = "llm"

    def __init__(self, model: str = "anthropic:claude-sonnet-5"):
        self.model = model
        self.name = f"llm:{model}"

    def extract(self, paper: Paper, text: str) -> DatasetFacts:
        from pydantic_ai import Agent

        agent = Agent(self.model, output_type=DatasetFacts, instructions=_INSTRUCTIONS)
        return agent.run_sync(f"TITLE: {paper.title}\n\nTEXT:\n{text}").output


class ScriptedFactsExtractor:
    name = "scripted"

    def __init__(self, fn: Callable[[Paper, str], DatasetFacts]):
        self.fn = fn

    def extract(self, paper: Paper, text: str) -> DatasetFacts:
        return self.fn(paper, text)
