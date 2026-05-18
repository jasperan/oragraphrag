import pytest

from oragraphrag.extract import EXTRACT_SCHEMA, ExtractionError, Extractor, validate_payload

GOOD = {
    "propositions": [
        {
            "text": "Vector indexes accelerate ANN search.",
            "triples": [
                {
                    "subject": "vector index",
                    "predicate": "accelerates",
                    "object": "ANN search",
                    "ontology_axis": "causal",
                    "confidence": 0.9,
                }
            ],
        }
    ]
}

BAD_AXIS = {
    "propositions": [
        {
            "text": "x",
            "triples": [
                {
                    "subject": "a",
                    "predicate": "b",
                    "object": "c",
                    "ontology_axis": "nonsense",
                    "confidence": 0.5,
                }
            ],
        }
    ]
}

MISSING_TRIPLES = {"propositions": [{"text": "x"}]}

BAD_CONFIDENCE = {
    "propositions": [
        {
            "text": "x",
            "triples": [
                {
                    "subject": "a",
                    "predicate": "b",
                    "object": "c",
                    "ontology_axis": "causal",
                    "confidence": 1.5,
                }
            ],
        }
    ]
}

EMPTY_PROPS = {"propositions": []}


def test_validate_accepts_good_payload():
    assert validate_payload(GOOD) == GOOD


def test_validate_accepts_empty_propositions_list():
    """An empty propositions array is a valid result — passage had nothing to extract."""
    assert validate_payload(EMPTY_PROPS) == EMPTY_PROPS


def test_validate_rejects_bad_axis():
    with pytest.raises(ExtractionError, match="axis"):
        validate_payload(BAD_AXIS)


def test_validate_rejects_missing_triples_key():
    with pytest.raises(ExtractionError):
        validate_payload(MISSING_TRIPLES)


def test_validate_rejects_confidence_outside_unit_interval():
    with pytest.raises(ExtractionError, match="confidence"):
        validate_payload(BAD_CONFIDENCE)


def test_validate_rejects_non_dict_top_level():
    with pytest.raises(ExtractionError):
        validate_payload("not a dict")
    with pytest.raises(ExtractionError):
        validate_payload({"wrong_key": []})


def test_schema_lists_all_five_ontology_axes():
    from oragraphrag.axes import ONTOLOGY_AXIS_NAMES

    enum = EXTRACT_SCHEMA["properties"]["propositions"]["items"]["properties"]["triples"][
        "items"
    ]["properties"]["ontology_axis"]["enum"]
    assert set(enum) == set(ONTOLOGY_AXIS_NAMES)
    assert len(enum) == 5


class _StubLLM:
    """Replays responses in order. Each .complete() consumes one."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.call_count = 0

    async def complete(self, prompt, *, schema=None, temperature=0.0):
        self.call_count += 1
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_extractor_passes_through_valid_payload():
    llm = _StubLLM(GOOD)
    ex = Extractor(llm)
    out = await ex.extract("some passage text")
    assert out == GOOD
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_extractor_retries_once_on_invalid_payload_and_succeeds():
    llm = _StubLLM(BAD_AXIS, GOOD)
    ex = Extractor(llm)
    out = await ex.extract("some passage text")
    assert out == GOOD
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_extractor_raises_after_second_failure():
    llm = _StubLLM(BAD_AXIS, BAD_AXIS)
    ex = Extractor(llm)
    with pytest.raises(ExtractionError):
        await ex.extract("some passage text")
    assert llm.call_count == 2  # one initial + one repair, then give up


@pytest.mark.asyncio
async def test_extractor_handles_llm_returning_string_json():
    """Some LLM backends return the JSON-encoded text instead of a parsed dict.
    Extractor must json.loads it before validating."""
    import json
    llm = _StubLLM(json.dumps(GOOD))
    ex = Extractor(llm)
    out = await ex.extract("some passage text")
    assert out == GOOD


@pytest.mark.asyncio
async def test_extractor_handles_string_with_malformed_json_first_then_good():
    """If the first response is a non-JSON string, repair-prompt should follow."""
    import json
    llm = _StubLLM("not json at all", json.dumps(GOOD))
    ex = Extractor(llm)
    out = await ex.extract("some passage text")
    assert out == GOOD
    assert llm.call_count == 2


def test_extract_template_renders_passage_into_prompt():
    """The Jinja template must inject the passage text verbatim."""
    from importlib.resources import files

    text = files("oragraphrag.prompts").joinpath("extract.j2").read_text()
    from jinja2 import Template

    rendered = Template(text).render(passage="Oracle 23ai introduced VECTOR_EMBEDDING.")
    assert "Oracle 23ai introduced VECTOR_EMBEDDING." in rendered
    # The axis rubric must enumerate all five axes by name.
    for axis in ("causal", "taxonomic", "temporal", "definitional", "exemplification"):
        assert axis in rendered.lower()
