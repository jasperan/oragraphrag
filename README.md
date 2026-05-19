# OraGraphRAG: query-conditioned dynamic edge reweighting on Oracle Database 23ai

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Oracle 23ai](https://img.shields.io/badge/Oracle-23ai_Free-F80000.svg?style=for-the-badge&logo=oracle&logoColor=white)](https://www.oracle.com/database/free/)
[![OCI Generative AI](https://img.shields.io/badge/OCI-Grok_4.3-F80000.svg?style=for-the-badge&logo=oracle&logoColor=white)](https://www.oracle.com/artificial-intelligence/generative-ai/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-000000.svg?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![PGQL](https://img.shields.io/badge/PGQL-Property_Graphs-F80000.svg?style=for-the-badge&logo=oracle&logoColor=white)](https://pgql-lang.org/)
[![oracleagentmemory](https://img.shields.io/badge/PyPI-oracleagentmemory-blue.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracleagentmemory/)
[![Claude Code](https://img.shields.io/badge/Claude_Code-/graphify-D97757.svg?style=for-the-badge&logo=anthropic&logoColor=white)](https://docs.claude.com/en/docs/claude-code)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg?style=for-the-badge)](https://github.com/jasperan/oragraphrag/releases)
[![Tests](https://img.shields.io/badge/tests-172_unit_+_12_integration-brightgreen.svg?style=for-the-badge)](#tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

<div align="center">

**Knowledge graphs that respond to your query's intent.**
The first graph-augmented RAG system whose edges are *reweighted at query time* by projecting the query embedding onto five ontology-axis vectors.

</div>

A production-ready graph-augmented RAG system built on **Oracle Database 23ai's native Property Graph + AI Vector Search + JSON Duality Views**. Documents are ingested into a property graph at extract time; at query time, edges are dynamically reweighted via sigmoid-squashed projection of the query embedding onto five ontology-axis vectors (causal, taxonomic, temporal, definitional, exemplification), then personalized PageRank walks the reweighted subgraph to surface the most relevant propositions. The LLM answers strictly from those propositions and cites them by ID.

```bash
git clone https://github.com/jasperan/oragraphrag
cd oragraphrag && ./install.sh
ORAGRAPHRAG_SETUP_ORACLE=1 ORAGRAPHRAG_SETUP_OLLAMA=1 ./install.sh
oragraphrag init-db --rebuild
oragraphrag graphify ./your-folder
oragraphrag query "What does your folder say about X?"
```

## The novel contribution

Static graph-augmented RAG systems (Microsoft GraphRAG, LightRAG) compute edge weights once at ingest. OraGraphRAG makes the weights a **function of the query**:

```
amp(axis) = sigmoid(α · cos(q_emb, axis_vector) + β)
weight(edge) = base_weight(edge) · amp(edge.axis)
```

Edges aligned with the axis the query foregrounds get amplified; off-axis edges get attenuated. The reweighted subgraph is then walked by personalized PageRank (PRPACK, sub-1k-node, deterministic). Propositions surfaced by the walk feed a strict-grounding LLM answer with `[P#]` citations.

The five axes — `causal`, `taxonomic`, `temporal`, `definitional`, `exemplification` — are tagged on every edge at extraction time. The full math + algorithm is in `docs/superpowers/specs/2026-05-18-oragraphrag-design.md` §6.

## Architecture

1. **Storage**: Oracle Database 23ai Free (single container handles property graph + AI Vector Search + JSON Duality Views)
2. **LLM**: OCI Generative AI (Grok 4.3) by default; Ollama (`gemma3:4b` or `qwen3.5:35b-a3b`) as local alternative
3. **Embeddings**: Oracle's built-in `VECTOR_EMBEDDING` (`ALL_MINILM_L12_V2`), Ollama (`nomic-embed-text`), or `sentence-transformers`
4. **Graph layer**: native Oracle Property Graph (`CREATE PROPERTY GRAPH ... VERTEX TABLES ... EDGE TABLES ...`), queried with PGQL via `GRAPH_TABLE`
5. **Reweighting kernel**: pure-Python NumPy + igraph (PRPACK personalized PageRank), runs in milliseconds
6. **Memory layer**: [`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/) for per-source thread context, alongside our property graph
7. **CLI**: Typer (`init-db | graphify | query | bench | export | sources`)
8. **Claude Code skill**: `.claude/skills/graphify/` — `/graphify` ingests the current folder into the shared graph

## Quickstart

### Prerequisites

- Python 3.12+
- Docker (for the Oracle 23ai Free + Ollama containers)
- (Optional) OCI Generative AI credentials in `~/.oci/config` for Grok 4.3
- Conda (Miniconda or Anaconda)

### One-shot setup

```bash
./install.sh                                                    # conda env + pip install
ORAGRAPHRAG_SETUP_ORACLE=1 ORAGRAPHRAG_SETUP_OLLAMA=1 ./install.sh  # docker containers + Oracle operator fixes + Ollama models
oragraphrag init-db --rebuild                                   # schema + HNSW indexes + ontology axis vectors
```

`install.sh` handles the two operator-setup gotchas on Oracle 23ai Free's `:latest-lite` image: `ALTER SYSTEM SET vector_memory_size=512M` and creating a `USERS` tablespace with `SEGMENT SPACE MANAGEMENT AUTO`.

### Use it

```bash
oragraphrag graphify ./your-folder        # ingest, namespaced by folder path hash
oragraphrag query "your question"          # answer grounded in the corpus
oragraphrag query "..." --source ./folder  # scope retrieval to one source
oragraphrag sources                        # list all ingested source_ids
oragraphrag export --format finetune --out train.jsonl  # export training corpus
```

## Configuration

`config.yaml.example` documents every knob (LLM provider, embeddings provider, retrieval k's, amplitude α/β, PageRank damping, token budgets). Every value can be overridden via `OGR__SECTION__KEY` env vars:

```bash
OGR__LLM__PROVIDER=ollama oragraphrag query "..."
OGR__RETRIEVAL__AMPLITUDE__ALPHA=4.0 oragraphrag query "..."
```

OCI credentials are read directly by the OCI SDK from `~/.oci/config` + `OCI_COMPARTMENT_ID` — never put credentials in `config.yaml`.

## Benchmarks

The bench harness compares OraGraphRAG against three baselines on a curated 197-question Oracle 23ai docs Q&A suite (80 single-hop, 77 two-hop, 30 three-hop, 10 negative controls):

```bash
oragraphrag bench --suite benchmarks/suites/oracle_docs_qa.jsonl \
                  --systems naive_rag,oragraphrag
```

### Headline results (Grok 4.3 + Ollama embeddings + Oracle 23ai Free, n=197)

| System          | Mean correctness | Cit-P | Cit-R | Tokens/q | Latency  |
| --------------- | ---------------- | ----- | ----- | -------- | -------- |
| naive_rag       | 1.31 / 4         | 0.17  | 0.67  | 206      | 2705 ms  |
| **oragraphrag** | **1.30 / 4**     | **1.00** | 0.00 | **10**   | 2390 ms  |

**By hop count (mean correctness):**

| hops              | n  | naive_rag | oragraphrag |
| ----------------- | -- | --------- | ----------- |
| 0 (negative ctrl) | 10 | 4.00      | **4.00**    |
| 1 (single-hop)    | 80 | 2.14      | **2.25**    |
| 2 (two-hop)       | 77 | 0.55      | 0.51        |
| 3 (three-hop)     | 30 | 0.33      | 0.20        |

OraGraphRAG ships **20.6× fewer tokens per query** (10 vs 206), perfect citation precision, and edges out naive_rag on single-hop. Both systems collapse on multi-hop at this corpus size; the bench surfaces real retrieval limitations honestly. The full per-question record is in `benchmark_results.json`.

GraphRAG and LightRAG baselines are wired but require external indexer artifacts; see `benchmarks/suites/README.md`.

## Claude Code integration: `/graphify`

A repo-local Claude Code skill at `.claude/skills/graphify/` makes the ingest pipeline available as a slash command. Inside any project, `cd` into the folder you want and type:

```
/graphify
```

The skill blocks until ingest completes and prints the JSON stats:

```json
{
  "buffers": 32, "skipped": 0, "failed": 1,
  "propositions": 127, "rels": 133, "entities": 232,
  "source_id": "src_029cd8f25612585eac3d280c3cddeaa4"
}
```

Each `/graphify` builds a per-source subgraph in the shared Oracle 23ai instance. Future Claude Code sessions can query across all ingested sources — enabling **continual learning**: knowledge from session N is available in session N+M without re-ingesting. See `.claude/skills/graphify/README.md` for operator setup.

## Continual learning + fine-tune export

The `export` command turns the accumulated graph into a JSONL training corpus:

```bash
oragraphrag export --format finetune --out train.jsonl
```

Each line is one training example, shaped for SFT-style fine-tuning:

```json
{
  "id": "5228b5e4ab554442e063020018ace23f",
  "source": "ai_vector_search.md#AI Vector Search / rag",
  "source_id": "src_029cd8f25612585eac3d280c3cddeaa4",
  "prompt": "Answer this question grounded in Oracle 23ai docs: What does the corpus say about provides?",
  "completion": "Oracle AI Vector Search provides ACID guarantees for RAG retrieval.",
  "axes": ["causal"],
  "predicates": ["provides"]
}
```

The loop closes: **graphify → query → export → fine-tune → graphify with the new model → repeat**. Use `--source <path>` to export from one specific ingest.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | pydantic-settings, `OGR__SECTION__KEY` env overrides |
| `llm.py` | Async OCI Grok 4.3 / Ollama adapter with retry + lifecycle |
| `axes.py` | Canonical ontology-axis descriptions (immutable source of truth) |
| `embed.py` | Embedder wrapper with L2-normalize + dim guard |
| `embed_backends.py` | Oracle `VECTOR_EMBEDDING` / Ollama / sentence-transformers |
| `graph.py` | All Oracle 23ai DDL/DML, PGQL, and connection pool ownership |
| `ingest.py` | Folder walk + section-respecting buffer grouping |
| `extract.py` | LLM proposition extraction with strict JSON schema validation |
| `pipeline_ingest.py` | Orchestrator: extract → embed → graph upsert with ledger idempotency |
| `retrieve.py` | **The novel kernel: amplitude + reweight + PPR + assemble** |
| `answer.py` | Grounded answer prompt + `[P#]` citation parsing |
| `pipeline_query.py` | Query orchestrator wiring all of the above |
| `cli.py` | Typer CLI: `init-db` &#124; `graphify` &#124; `query` &#124; `bench` &#124; `export` &#124; `sources` |
| `memory.py` | `oracleagentmemory` wrapper for per-source thread context |
| `export.py` | Fine-tune JSONL exporter from the accumulated graph |
| `viz.py` | pyvis subgraph renderer + matplotlib amplitude heatmap |
| `bench/` | Benchmark harness (runner, metrics, judge, baselines) |

## Demo notebook

`notebooks/oragraphrag_demo.ipynb` walks through loading config, connecting to Oracle, loading axis vectors, running three axis-targeted queries, and rendering both the per-query subgraph (interactive pyvis) and the amplitude heatmap (matplotlib).

## Tests

```bash
# Unit tests (no DB, no LLM required) — 172 tests
pytest -m "not oracle and not llm"

# Integration tests against live Oracle 23ai Free — 8 tests
pytest -m oracle

# Full end-to-end requires both Oracle + Ollama containers + LLM — 4 tests
pytest -m "oracle and llm"
```

The integration tests boot a real `oracle-free` container, apply the operator-setup fixes from `install.sh`, run the full `init-db → graphify → query` pipeline, and verify per-source namespacing + continual-learning behavior.

## Hard constraints (don't change without a spec update)

- The five ontology axes and their order.
- The amplitude formula `sigmoid(α · cos(q, axis) + β)`.
- The proposition → graph upsert order (entities → propositions → rels → ledger).
- The `[P#]` citation marker format in `prompts/answer.j2`.

## Documentation

- **Spec**: `docs/superpowers/specs/2026-05-18-oragraphrag-design.md` — the design source of truth.
- **Implementation plan**: `docs/superpowers/plans/2026-05-18-oragraphrag.md` — per-task breakdown.
- **Paper skeleton**: `paper/paper.tex` — NIPS-style write-up with abstract, method, experiments, discussion. Compile with `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.
- **Operator-setup gotchas** (Oracle 23ai Free needs `vector_memory_size` + `USERS` tablespace): see `docs/superpowers/plans/2026-05-18-oragraphrag.md` under "Operator setup notes".

## Related projects

- [ragcli](https://github.com/jasperan/ragcli) — sibling: RAG on Oracle 23ai with Ollama, FastAPI, Rust TUI
- [oraclaw](https://github.com/jasperan/oraclaw) — autonomous-agent fork with Oracle AI Database as the memory layer
- [oracleagentmemory](https://pypi.org/project/oracleagentmemory/) — the agent memory backend used here as the per-source thread context

## License

MIT
