import pytest

from oragraphrag.answer import Answerer, AnswerResult, Citation


class _StubLLM:
    """Returns a pre-canned response, ignoring the prompt."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    async def complete(self, prompt, *, schema=None, temperature=0.0):
        self.last_prompt = prompt
        return self.response


@pytest.mark.asyncio
async def test_answer_renders_propositions_and_extracts_citations():
    llm = _StubLLM("Vector indexes accelerate ANN search. [P1]")
    a = Answerer(llm=llm, token_budget=4000)
    result = await a.answer(
        question="What do vector indexes do?",
        propositions=[
            {
                "id": b"\x01",
                "text": "Vector indexes accelerate ANN search.",
                "source_doc": "concepts.md",
                "source_span": "Vector Indexes",
            },
        ],
    )
    assert isinstance(result, AnswerResult)
    assert "ANN search" in result.text
    assert len(result.citations) == 1
    assert result.citations[0].proposition_id == b"\x01"
    assert result.citations[0].source_doc == "concepts.md"
    assert result.citations[0].source_span == "Vector Indexes"


@pytest.mark.asyncio
async def test_answer_returns_no_info_message_when_propositions_empty():
    llm = _StubLLM("should not be called")
    a = Answerer(llm=llm, token_budget=4000)
    result = await a.answer(question="What is X?", propositions=[])
    assert "don't have information" in result.text.lower()
    assert result.citations == []
    # The LLM must NOT have been called when propositions is empty.
    assert llm.last_prompt is None


@pytest.mark.asyncio
async def test_answer_maps_multiple_citations_in_order():
    llm = _StubLLM("First [P1]. Second [P2]. Third [P1] again.")
    a = Answerer(llm=llm, token_budget=4000)
    result = await a.answer(
        question="Multi-prop?",
        propositions=[
            {"id": b"\x01", "text": "first", "source_doc": "d1.md", "source_span": "s1"},
            {"id": b"\x02", "text": "second", "source_doc": "d2.md", "source_span": "s2"},
        ],
    )
    # All three citation markers map back.
    assert len(result.citations) == 3
    assert [c.proposition_id for c in result.citations] == [b"\x01", b"\x02", b"\x01"]


@pytest.mark.asyncio
async def test_answer_ignores_out_of_range_citation_marker():
    """[P99] when only 2 propositions provided must NOT produce a citation."""
    llm = _StubLLM("Cited but bogus [P99].")
    a = Answerer(llm=llm, token_budget=4000)
    result = await a.answer(
        question="Q?",
        propositions=[
            {"id": b"\x01", "text": "t1", "source_doc": "d", "source_span": "s"},
            {"id": b"\x02", "text": "t2", "source_doc": "d", "source_span": "s"},
        ],
    )
    assert result.citations == []


@pytest.mark.asyncio
async def test_answer_handles_proposition_id_as_hex_string():
    """Some callers pass proposition ids as hex strings (from JSON columns)."""
    llm = _StubLLM("Quoted [P1].")
    a = Answerer(llm=llm, token_budget=4000)
    result = await a.answer(
        question="Q?",
        propositions=[
            {"id": b"\xaa\xbb".hex(), "text": "t", "source_doc": "d", "source_span": "s"},
        ],
    )
    assert len(result.citations) == 1
    assert result.citations[0].proposition_id == b"\xaa\xbb"


@pytest.mark.asyncio
async def test_answer_prompt_includes_proposition_markers_and_question():
    llm = _StubLLM("answer")
    a = Answerer(llm=llm, token_budget=4000)
    await a.answer(
        question="What is HNSW?",
        propositions=[
            {
                "id": b"\x01",
                "text": "HNSW is a vector index.",
                "source_doc": "c.md",
                "source_span": "Indexes",
            },
        ],
    )
    assert llm.last_prompt is not None
    assert "What is HNSW?" in llm.last_prompt
    assert "HNSW is a vector index." in llm.last_prompt
    assert "[P1]" in llm.last_prompt


def test_citation_dataclass_fields():
    c = Citation(proposition_id=b"\x01", source_doc="d.md", source_span="s")
    assert c.proposition_id == b"\x01"
    assert c.source_doc == "d.md"
    assert c.source_span == "s"


def test_answer_result_dataclass_fields():
    r = AnswerResult(text="hi", citations=[])
    assert r.text == "hi"
    assert r.citations == []


def test_answer_template_renders_no_info_marker_distinguishable():
    """The 'I don't have information' string must be the exact one Answerer
    returns when propositions is empty — keeps the contract checkable."""
    from oragraphrag.answer import _NO_INFO

    assert "don't have information" in _NO_INFO.lower()
