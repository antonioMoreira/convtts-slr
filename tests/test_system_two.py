from convtts_slr.models import DatasetFacts, Paper, Quoted
from convtts_slr.system_two import LLMFactsExtractor, ScriptedFactsExtractor


def test_scripted_facts_extractor_delegates_to_its_function():
    seen = {}

    def fn(paper: Paper, text: str) -> DatasetFacts:
        seen["paper"] = paper
        seen["text"] = text
        return DatasetFacts(total_hours=Quoted(value=5.0))

    extractor = ScriptedFactsExtractor(fn)
    p = Paper(id="p1", title="t")
    result = extractor.extract(p, "some text")
    assert extractor.name == "scripted"
    assert seen == {"paper": p, "text": "some text"}
    assert result.total_hours.value == 5.0


def test_llm_facts_extractor_name_reflects_its_model():
    assert LLMFactsExtractor().name == "llm:anthropic:claude-sonnet-5"
    assert LLMFactsExtractor(model="anthropic:claude-x").name == "llm:anthropic:claude-x"
