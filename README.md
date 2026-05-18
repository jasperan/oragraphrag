# OraGraphRAG

Oracle-backed graph-augmented RAG with query-conditioned dynamic edge reweighting.

OraGraphRAG ingests a folder of documents into an Oracle Database 23ai property
graph, then answers questions by walking a bounded subgraph whose edge weights
are recomputed per query based on which ontology axis the query is talking
about (causal, taxonomic, temporal, definitional, or exemplification).

**Status:** Early scaffold — implementation in progress. See
`docs/superpowers/plans/2026-05-18-oragraphrag.md` for the implementation
plan and `docs/superpowers/specs/2026-05-18-oragraphrag-design.md` for the
design spec. Full quickstart will land in Task 18.
