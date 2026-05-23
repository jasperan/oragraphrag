"""Plain data contracts for atomic ingest writes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestEntity:
    name: str
    kind: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class IngestTriple:
    subject: str
    predicate: str
    object: str
    ontology_axis: str
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestProposition:
    text: str
    embedding: list[float]
    triples: tuple[IngestTriple, ...]


@dataclass(frozen=True, slots=True)
class IngestUnit:
    doc_id: str
    section_path: str
    span_hashes: tuple[str, ...]
    source_id: str
    entities: tuple[IngestEntity, ...]
    propositions: tuple[IngestProposition, ...]


@dataclass(frozen=True, slots=True)
class IngestWriteStats:
    entities: int = 0
    propositions: int = 0
    rels: int = 0

