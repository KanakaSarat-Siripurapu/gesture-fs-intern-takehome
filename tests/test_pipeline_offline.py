"""
Offline unit tests for the Q&A pipeline.

These complement tests/test_pipeline.py: they use stub doubles for the vector
store and the LLM, so they run in milliseconds and need no model downloads.
They cover the parts of ask_question() that a model-backed test can't pin down
deterministically: prompt construction, context bounding, and edge cases.

Run: pytest tests/test_pipeline_offline.py -v
"""

from typing import List

import pytest

from src.pipeline import (
    MAX_CONTEXT_CHARS,
    NO_ANSWER,
    PROMPT_TEMPLATE,
    TOP_K,
    _build_context,
    ask_question,
    format_result,
)


class StubDoc:
    def __init__(self, page_content: str):
        self.page_content = page_content


class StubVectorStore:
    """Records the search it was asked for and returns canned chunks."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.calls = []

    def similarity_search(self, query, k=4):
        self.calls.append((query, k))
        return [StubDoc(c) for c in self.chunks[:k]]


class StubLLM:
    """Records the prompt it received and returns a fixed answer."""

    def __init__(self, answer: str = "The Starter package is $2,500/month."):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return [{"generated_text": self.answer}]


@pytest.fixture
def store():
    return StubVectorStore(
        [
            "STARTER PACKAGE - $2,500/month. Best for small businesses.",
            "GROWTH PACKAGE - $5,500/month. Best for scaling businesses.",
            "ENTERPRISE PACKAGE - $12,000/month. Dedicated team.",
            "PAYMENT TERMS - billed monthly on the 1st.",
        ]
    )


class TestRetrievalWiring:
    def test_requests_top_k_chunks(self, store):
        ask_question(store, StubLLM(), "How much is Starter?")
        assert store.calls == [("How much is Starter?", TOP_K)]

    def test_sources_are_the_retrieved_chunks(self, store):
        result = ask_question(store, StubLLM(), "How much is Starter?")
        assert result["sources"] == store.chunks[:TOP_K]


class TestPromptConstruction:
    def test_prompt_contains_question_and_context(self, store):
        llm = StubLLM()
        ask_question(store, llm, "How much is Starter?")
        prompt = llm.prompts[0]
        assert "How much is Starter?" in prompt
        assert "$2,500/month" in prompt
        assert prompt.startswith(PROMPT_TEMPLATE.split("{")[0])

    def test_question_survives_a_huge_context(self, store):
        """The question sits at the end of the prompt; it must not be crowded out."""
        big = StubVectorStore(["x" * 5000 for _ in range(TOP_K)])
        llm = StubLLM()
        ask_question(big, llm, "How much is Starter?")
        prompt = llm.prompts[0]
        assert "How much is Starter?" in prompt
        assert len(prompt) < MAX_CONTEXT_CHARS + len(PROMPT_TEMPLATE) + 200


class TestContextBuilder:
    def test_joins_chunks(self):
        assert _build_context(["a", "b"]) == "a\n\nb"

    def test_skips_blank_chunks(self):
        assert _build_context(["a", "   ", "b"]) == "a\n\nb"

    def test_respects_the_char_budget(self):
        assert len(_build_context(["a" * 100, "b" * 100], max_chars=50)) <= 50

    def test_keeps_the_best_match_first(self):
        assert _build_context(["best", "worst"], max_chars=4) == "best"


class TestAnswerHandling:
    def test_answer_is_the_generated_text(self, store):
        result = ask_question(store, StubLLM("$2,500/month."), "How much?")
        assert result["answer"] == "$2,500/month."

    def test_whitespace_is_stripped(self, store):
        result = ask_question(store, StubLLM("  padded  "), "How much?")
        assert result["answer"] == "padded"

    def test_empty_generation_falls_back(self, store):
        result = ask_question(store, StubLLM("   "), "How much?")
        assert result["answer"] == NO_ANSWER

    def test_empty_question_does_not_call_the_llm(self, store):
        llm = StubLLM()
        result = ask_question(store, llm, "   ")
        assert result["sources"] == []
        assert llm.prompts == []

    def test_no_retrieved_chunks_falls_back(self):
        llm = StubLLM()
        result = ask_question(StubVectorStore([]), llm, "How much?")
        assert result == {"answer": NO_ANSWER, "sources": []}
        assert llm.prompts == []


class TestFormatting:
    def test_shows_sources_and_answer(self, store):
        text = format_result(ask_question(store, StubLLM(), "How much?"))
        assert "📄 Sources:" in text
        assert "1. STARTER PACKAGE" in text
        assert "💬 Answer: The Starter package is $2,500/month." in text

    def test_handles_no_sources(self):
        text = format_result({"answer": NO_ANSWER, "sources": []})
        assert "(none)" in text
