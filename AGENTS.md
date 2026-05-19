# CLAUDE.md — OraGraphRAG agent guidance

This is the agent-facing guide for OraGraphRAG. Read it before touching the codebase.

## Project shape

OraGraphRAG is an Oracle-backed graph-augmented RAG system. The novel contribution is **query-conditioned dynamic edge reweighting**: at query time, edges are reweighted by sigmoid-squashed projection of the query embedding onto five ontology-axis vectors (causal, taxonomic, temporal, definitional, exemplification), and personalized PageRank walks the reweighted bounded subgraph.

The spec lives at `docs/superpowers/specs/2026-05-18-oragraphrag-design.md`. The implementation plan lives at `docs/superpowers/plans/2026-05-18-oragraphrag.md`.

## Hard constraints

- **Oracle Database 23ai is the storage layer.** Property Graph + AI Vector Search + JSON Duality Views. Never replace it.
- **The five ontology axes are fixed.** `causal`, `taxonomic`, `temporal`, `definitional`, `exemplification` (in that order). Changing names or count requires `oragraphrag init-db --rebuild` because the stored axis vectors will shift.
- **LLM layer is routable.** OCI Generative AI Grok 4.3 by default; Ollama is the local alternative. The `LLM` adapter in `src/oragraphrag/llm.py` is async-first; backends conform to a narrow Protocol.
- **The graph is the only DB-touching module.** `src/oragraphrag/graph.py` owns every `oracledb` call. Other modules take/return plain Python types.

## Test conventions

- `pytest -m "not oracle and not llm"` — the default; 151+ tests, all in-memory.
- `pytest -m oracle` — integration tests against a live Oracle 23ai Free container (6 tests in `tests/test_graph.py`).
- `pytest -m "oracle and llm"` — full end-to-end via the CLI against live Oracle + Ollama (4 tests in `tests/integration/`).
- Autouse fixture `env_no_oracle` strips `OGR__*` and `ORACLE_*` env vars before every test.

## Common workflows

```bash
# Install
./install.sh

# Bring up infra
ORAGRAPHRAG_SETUP_ORACLE=1 ORAGRAPHRAG_SETUP_OLLAMA=1 ./install.sh

# Initialize schema (creates tables, HNSW indexes, property graph, axis vectors)
oragraphrag init-db --rebuild

# Ingest
oragraphrag graphify ./your-folder

# Query
oragraphrag query "What does the VECTOR datatype store?"

# Benchmark
oragraphrag bench --suite benchmarks/suites/oracle_docs_qa.jsonl --systems naive_rag,oragraphrag
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | pydantic-settings, env-var overrides via `OGR__SECTION__KEY` |
| `llm.py` | LLM adapter (oci_grok, ollama backends) with retry + lifecycle |
| `axes.py` | Canonical ontology-axis descriptions (fixed source of truth) |
| `embed.py` | Embedder wrapper with L2-normalize + dim guard |
| `embed_backends.py` | Concrete embedders (Oracle, Ollama, sentence-transformers) |
| `graph.py` | All Oracle 23ai DDL/DML, PGQL, and connection pool ownership |
| `ingest.py` | Folder walk + section-respecting buffer grouping |
| `extract.py` | LLM proposition extraction with strict JSON schema validation |
| `pipeline_ingest.py` | Orchestrator: extract → embed → graph upsert with ledger idempotency |
| `retrieve.py` | The novel kernel: amplitude + reweight + PPR + assemble |
| `answer.py` | Grounded answer prompt + [P#] citation parsing |
| `pipeline_query.py` | Query orchestrator wiring all of the above |
| `cli.py` | Typer CLI (`init-db | graphify | query | bench`) |
| `viz.py` | pyvis subgraph + matplotlib amplitude heatmap |
| `bench/` | Benchmark harness (runner, metrics, judge, baselines) |

## Things NOT to change without a spec update

- The five ontology axis names or their order in `axes.py`.
- The amplitude formula `sigmoid(α · cos(q, axis) + β)`.
- The base_weight + amplitude product in `reweight_edges`.
- The proposition → graph upsert order (entities first, then propositions, then rels, then ledger).
- The `[P#]` citation marker format in `answer.j2`.

## Operator setup gotchas

Oracle 23ai Free's `:latest-lite` image does NOT come up ready for VECTOR + HNSW. Two fixes are required (handled by `install.sh` when `ORAGRAPHRAG_SETUP_ORACLE=1`):

1. `ALTER SYSTEM SET vector_memory_size=512M SCOPE=SPFILE;` (then restart).
2. A USERS tablespace with `SEGMENT SPACE MANAGEMENT AUTO` (the default SYSTEM tablespace doesn't support VECTOR).

Plus: the `ORAGRAPH` user needs `CREATE CONNECT, RESOURCE, UNLIMITED TABLESPACE, CREATE PROPERTY GRAPH` grants.

## License

MIT.
