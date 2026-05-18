"""Proposition extraction + strict JSON schema validation.

The Extractor renders the extract.j2 prompt with the passage text, calls the
LLM with a schema hint, validates the returned payload, and retries once
with a stricter "fix this JSON" repair prompt before giving up.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jinja2 import Template

from oragraphrag.axes import ONTOLOGY_AXIS_NAMES


class ExtractionError(RuntimeError):
    """Raised when the LLM cannot produce a schema-conforming payload."""


EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["propositions"],
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "triples"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "triples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "subject",
                                "predicate",
                                "object",
                                "ontology_axis",
                                "confidence",
                            ],
                            "properties": {
                                "subject": {"type": "string", "minLength": 1},
                                "predicate": {"type": "string", "minLength": 1},
                                "object": {"type": "string", "minLength": 1},
                                "ontology_axis": {"enum": list(ONTOLOGY_AXIS_NAMES)},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                },
            },
        }
    },
}


def validate_payload(payload: Any) -> dict[str, Any]:
    """Strict schema validation. Raises ExtractionError on any deviation.

    Tighter than a generic JSON-schema lib so error messages name the exact
    failing key — that's how Task 8 will route a bad chunk to the failure log.
    """
    if not isinstance(payload, dict) or "propositions" not in payload:
        raise ExtractionError("payload is not a dict with a 'propositions' key")
    props = payload["propositions"]
    if not isinstance(props, list):
        raise ExtractionError("'propositions' must be a list")
    for i, p in enumerate(props):
        if not isinstance(p, dict):
            raise ExtractionError(f"proposition[{i}] is not a dict")
        if "text" not in p or "triples" not in p:
            raise ExtractionError(
                f"proposition[{i}] missing 'text' or 'triples'"
            )
        if not isinstance(p["text"], str) or not p["text"].strip():
            raise ExtractionError(f"proposition[{i}].text must be non-empty string")
        if not isinstance(p["triples"], list):
            raise ExtractionError(f"proposition[{i}].triples must be a list")
        for j, t in enumerate(p["triples"]):
            if not isinstance(t, dict):
                raise ExtractionError(f"proposition[{i}].triples[{j}] is not a dict")
            for key in ("subject", "predicate", "object", "ontology_axis", "confidence"):
                if key not in t:
                    raise ExtractionError(
                        f"proposition[{i}].triples[{j}] missing key: {key}"
                    )
            if t["ontology_axis"] not in ONTOLOGY_AXIS_NAMES:
                raise ExtractionError(
                    f"proposition[{i}].triples[{j}] invalid axis: {t['ontology_axis']!r}"
                )
            conf_raw = t["confidence"]
            if isinstance(conf_raw, bool) or not isinstance(conf_raw, (int, float)):
                raise ExtractionError(
                    f"proposition[{i}].triples[{j}].confidence is not a number"
                )
            conf = float(conf_raw)
            if not (0.0 <= conf <= 1.0):
                raise ExtractionError(
                    f"proposition[{i}].triples[{j}].confidence out of [0, 1]: {conf}"
                )
    return payload


def _strip_markdown_fence(s: str) -> str:
    """Strip a single ```[lang]?\n ... ``` fence if present.

    Many LLMs wrap JSON responses in markdown fences even when told not to.
    Defensive: cheaper to strip than to pay for a repair round-trip.
    """
    text = s.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence and optional language tag.
    first_newline = text.find("\n")
    if first_newline == -1:
        return text[3:].rstrip("`").strip()
    text = text[first_newline + 1 :]
    # Drop the trailing fence.
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text


def _parse_json_string(s: str) -> Any:
    cleaned = _strip_markdown_fence(s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"LLM returned invalid JSON string: {e}") from e


class Extractor:
    """Renders the prompt, calls the LLM, validates, retries once on failure."""

    def __init__(self, llm: Any, template_text: str | None = None) -> None:
        self._llm = llm
        text = template_text or (
            files("oragraphrag.prompts").joinpath("extract.j2").read_text()
        )
        self._tmpl = Template(text)

    async def _call_and_validate(self, prompt: str) -> dict[str, Any]:
        payload = await self._llm.complete(
            prompt, schema=EXTRACT_SCHEMA, temperature=0.0
        )
        if isinstance(payload, str):
            payload = _parse_json_string(payload)
        return validate_payload(payload)

    async def extract(self, passage: str) -> dict[str, Any]:
        prompt = self._tmpl.render(passage=passage)
        # First attempt with schema hint (lets backends use format=json).
        try:
            return await self._call_and_validate(prompt)
        except ExtractionError:
            pass
        # Repair attempt: stricter instruction reminding the LLM of the schema.
        repair = (
            "Your previous response was invalid JSON or violated the schema. "
            "Respond ONLY with valid JSON matching the schema above. No prose, "
            "no markdown fences, just the JSON object.\n\n"
            f"{prompt}"
        )
        return await self._call_and_validate(repair)
