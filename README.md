# OraGraphRAG

Oracle-backed graph-augmented RAG with **query-conditioned dynamic edge reweighting**.

Documents are ingested into an Oracle Database 23ai property graph at ingest
time. At query time, edge weights are recomputed via sigmoid-squashed
projection of the query embedding onto five ontology-axis vectors (causal,
taxonomic, temporal, definitional, exemplification), then personalized
PageRank walks the reweighted subgraph to surface the most relevant
propositions. The LLM answers strictly from those propositions and cites
them by ID.

The full stack runs on Oracle 23ai's native Property Graph + AI Vector
Search + JSON Duality Views, with the LLM layer routable to OCI Generative
AI (Grok 4.3) or local Ollama.

## Quickstart

```bash
# 1. Set up conda env + install
./install.sh

# 2. (Optional) bring up Oracle + Ollama containers via Docker
ORAGRAPHRAG_SETUP_ORACLE=1 ORAGRAPHRAG_SETUP_OLLAMA=1 ./install.sh

# 3. Initialize the schema
oragraphrag init-db --rebuild

# 4. Ingest a folder of documents
oragraphrag graphify ./your-folder

# 5. Ask a question
oragraphrag query "What does the VECTOR datatype store?"
```

## Configuration

`config.yaml.example` documents every knob. Override any value via
`OGR__SECTION__KEY` environment variables, e.g.:

```bash
OGR__LLM__PROVIDER=ollama oragraphrag query "..."
OGR__RETRIEVAL__AMPLITUDE__ALPHA=4.0 oragraphrag query "..."
```

Common choices:
- **LLM**: `oci_grok` (default) or `ollama`
- **Embeddings**: `oracle` (default; uses Oracle 23ai's `VECTOR_EMBEDDING`),
  `ollama`, or `sentence_transformers`

## Benchmarks

The benchmark harness compares OraGraphRAG against naive RAG, Microsoft
GraphRAG, and LightRAG on a 200-question Oracle docs Q&A suite (80
single-hop, 80 two-hop, 40 three-hop):

```bash
oragraphrag bench \
  --suite benchmarks/suites/oracle_docs_qa.jsonl \
  --systems naive_rag,oragraphrag
```

GraphRAG and LightRAG runners are wired but require their own indexer
artifacts; see `benchmarks/suites/README.md` for the curation checklist.

## Architecture

The spec lives at `docs/superpowers/specs/2026-05-18-oragraphrag-design.md`.
Five ontology axes with canonical descriptions seed the static axis
vectors; per-query amplitude modulation runs in pure-Python NumPy with
igraph PRPACK personalized PageRank over a bounded PGQL subgraph.

## Tests

```bash
# Unit tests (no DB, no LLM required)
pytest -m "not oracle and not llm"

# Integration tests against live Oracle 23ai Free
pytest -m oracle

# Full end-to-end (requires both Oracle + Ollama containers)
pytest -m "oracle and llm"
```

## Claude Code integration

The project ships with a `/graphify` slash command for Claude Code. Type `/graphify` in any Claude Code session to ingest the current working directory into the shared OraGraphRAG Oracle 23ai graph. See `.claude/skills/graphify/README.md` for operator setup and configuration.

## Status

Tasks 1-21 of the implementation plan complete. See
`docs/superpowers/plans/2026-05-18-oragraphrag.md` for the per-task
breakdown.

## License

MIT
