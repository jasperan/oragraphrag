# OraGraphRAG — Design Spec

**Date:** 2026-05-18
**Status:** Approved by user, ready for implementation planning
**Project:** OraGraphRAG — Oracle-backed graph-augmented RAG with query-conditioned dynamic edge reweighting

---

## 1. Purpose and contribution

OraGraphRAG is an Oracle showcase + research demo. It demonstrates Oracle AI Database 23ai's combined Property Graph, AI Vector Search, and JSON Duality View capabilities as the backing store for a graph-augmented RAG system that introduces one novel technique:

- **Query-conditioned dynamic edge reweighting via ontology-axis projection.** Edges in the knowledge graph are tagged at extraction time with an ontology axis (causal, taxonomic, temporal, definitional, exemplification). At query time, the query embedding is projected onto each axis vector to produce a per-axis amplitude scalar that modulates edge weights before spreading activation runs over a bounded subgraph.

Two contributions are reported in the paper:

1. **The reweighting technique** (the research angle).
2. **Oracle AI Database 23ai as a unified PGQL + Vector + JSON backend** (the marketing angle).

Multi-ontology layered graphs (separate per-axis graphs) and conversation-aware spreading activation are explicitly out of scope for v1; both are listed as future work.

## 2. Inputs that shaped this design

User-confirmed choices during brainstorming:

- Project category: Oracle showcase + research demo (paper + notebook + dockerized stack), CLI + notebook + paper.pdf surface, no web UI in v1.
- LLM layer: OCI Generative AI service with Grok 4.3 as default, Ollama (Qwen3.5-35B-A3B) as local fallback, following the Oracle AI Developer Hub "Choose Your Path" pattern.
- Amplitude interpretation: query-conditioned activation spreading. Base weights from co-occurrence + embedding similarity; sigmoid-squashed projection of the query onto edge ontology axes modulates the base weight; personalized PageRank walks the reweighted subgraph.
- Demo corpus: Oracle Database 23ai documentation.
- Ingest strategy: proposition-level extraction with the LLM producing `(subject, predicate, object, ontology_axis)` tuples.
- Graph storage: Oracle 23ai native Property Graph (PGQL) + AI Vector Search (VECTOR + HNSW indexes). JSON Duality Views expose query-time subgraphs to the answer prompt.
- Surface: `oragraphrag` CLI (Graphify-style `graphify <folder>` UX) + Jupyter notebook + `paper.tex`.
- Baselines: naive RAG, Microsoft GraphRAG, LightRAG, on a 200-question Oracle 23ai docs Q&A set.
- Architectural approach: Approach A — two-stage retrieve-then-reweight (vector seed → PGQL subgraph → in-Python reweighting → igraph PageRank → JSON Duality View → LLM answer).

## 3. Architecture and components

OraGraphRAG is a Python package + CLI. Modules under `src/oragraphrag/`:

| Module | Purpose | Key dependency |
|---|---|---|
| `oragraphrag.ingest` | Walks a folder (md/pdf/code), produces normalized text spans | `pypdf`, `tree-sitter`, `markdown-it-py` |
| `oragraphrag.extract` | LLM proposition extraction → `(s, p, o, ontology_axis)` tuples | OCI GenAI Grok 4.3 / Ollama Qwen3.5-35B-A3B |
| `oragraphrag.embed` | Embeds propositions + entities; loads static ontology-axis vectors | Oracle 23ai `VECTOR_EMBEDDING` or local model |
| `oragraphrag.graph` | Oracle 23ai property-graph DDL/DML + PGQL queries | `oracledb` |
| `oragraphrag.retrieve` | Vector seed → PGQL subgraph → reweighting → spreading activation | `igraph`, `numpy` |
| `oragraphrag.answer` | Assembles top propositions via JSON Duality View → LLM → answer + citations | OCI GenAI / Ollama |
| `oragraphrag.bench` | Eval harness against naive RAG / GraphRAG / LightRAG baselines | LLM-as-judge with Grok 4.3 |
| `oragraphrag.cli` | `oragraphrag graphify | query | bench | init-db` | `typer` |
| `oragraphrag.llm` | Thin adapter (`complete`, `embed`) over OCI GenAI / Ollama / OpenAI-compatible | `oci`, `httpx` |

External surfaces:

- `oragraphrag graphify <folder>` — one-command ingest.
- `oragraphrag query "..."` — single shot or REPL.
- `oragraphrag bench --suite oracle-docs-qa --systems all`.
- `oragraphrag init-db [--rebuild]` — DDL + HNSW indexes + property graph creation.
- `notebooks/oragraphrag_demo.ipynb` — visualizes the reweighted subgraph per query with `pyvis`.
- `paper/paper.tex` — NIPS-style write-up with benchmark numbers.

Repo layout mirrors `ragcli`:

```
oragraphrag/
  README.md, CLAUDE.md, AGENTS.md, GEMINI.md
  pyproject.toml, requirements.txt
  docker-compose.yml          # Oracle 23ai Free + the app
  Dockerfile
  install.sh
  config.yaml.example
  src/oragraphrag/
  notebooks/oragraphrag_demo.ipynb
  benchmarks/
    suites/oracle_docs_qa.jsonl
    baselines/{naive_rag,graphrag,lightrag,oragraphrag}/
    configs/
  paper/paper.tex
  docs/superpowers/specs/
  tests/
```

Module boundaries:

- `oragraphrag.graph` is the only module that talks to Oracle. Every other module takes/returns plain Python types.
- `oragraphrag.extract` and `oragraphrag.answer` are the only modules that call an LLM, both via `oragraphrag.llm`.
- `oragraphrag.retrieve` is pure compute over a subgraph returned from `oragraphrag.graph`; trivially unit-testable with a hand-built igraph.

## 4. Graph schema

Oracle 23ai property graph `oragraph` over four tables.

**Nodes:**

- `Entity(id PK, name, kind, embedding VECTOR(<dim>, FLOAT32), mention_count, created_at)`
  - `kind` is a free-form string set by the extractor (`feature`, `component`, `parameter`, `error_code`, `version`, `concept`, …); not a fixed enum.
- `Proposition(id PK, text, source_doc, source_span, embedding VECTOR(<dim>, FLOAT32), created_at)`
  - One row per atomic claim extracted from the corpus.

**Edges:**

- `MENTIONS` (`Proposition → Entity`): `role ∈ {subject, object}`, `confidence FLOAT`.
- `REL` (`Entity → Entity`):
  - `predicate VARCHAR2(128)` — surface predicate from the extractor.
  - `ontology_axis VARCHAR2(32)` — one of `{causal, taxonomic, temporal, definitional, exemplification}`. Resolved per the disagreement rule in §5 step 5.
  - `base_weight FLOAT` — `α · cooc_pmi + β · sem_sim`, normalized to [0, 1].
  - `support_propositions JSON` — proposition IDs that asserted this edge.
  - `support_axis_counts JSON` — per-axis vote counts for resolving `ontology_axis`.
  - `created_at`, `last_seen_at`.

**Ontology axes:** `OntologyAxis(name PK, description, axis_embedding VECTOR(<dim>))`. Axis vectors are produced once at `init-db` by embedding a short canonical description per axis (e.g., causal → "X causes Y; X leads to Y; X is the reason for Y; …"). Stored once, used at every query.

**Indexes:**

- HNSW on `Entity.embedding` and `Proposition.embedding`.
- B-tree on `REL(ontology_axis)`, `REL(predicate)`.
- `CREATE PROPERTY GRAPH oragraph` over the four tables.

**Embedding dim** is configurable but locked at install time. Switching providers later requires `init-db --rebuild`.

## 5. Ingest pipeline

`oragraphrag graphify <folder>` runs:

1. **Walk and normalize.** Recurse the folder, route by extension (md, pdf, py/ts/sql via tree-sitter, plain text). Emit `Span(doc_id, section_path, text)`.
2. **Span buffering.** Group spans by section, max ~1200 tokens per buffer, 100-token rolling overlap only at section boundaries.
3. **Proposition extraction.** Call the LLM (Grok 4.3 default) with `prompts/extract.j2`, structured-output JSON:
   ```json
   {"propositions": [
     {"text": "...", "triples": [
       {"subject": "...", "predicate": "...", "object": "...",
        "ontology_axis": "causal|taxonomic|temporal|definitional|exemplification",
        "confidence": 0.0}
     ]}
   ]}
   ```
   Concurrency bounded (default 8); exponential backoff on retries.
4. **Entity canonicalization.** For each new entity surface form, embed and search existing `Entity` rows with `VECTOR_DISTANCE` above 0.92 cosine; merge to canonical node if found, otherwise insert.
5. **Edge upserts.** `MERGE` `REL` rows on `(s, o, predicate)`; recompute `base_weight` incrementally; append the proposition id to `support_propositions`. If extractions disagree on `ontology_axis` for the same `(s, o, predicate)`, the most-frequent axis across `support_propositions` wins; ties broken by most-recent assertion. Axis vote counts are tracked in a `support_axis_counts JSON` column on `REL`.
6. **Embedding pass.** Batch-embed missing embeddings via `UPDATE … SET embedding = :vec`.
7. **Index refresh.** Refresh HNSW indexes if more than `N` new rows were inserted (default `N=1000`).

**Idempotency.** An `IngestLedger` table tracks span hashes; re-running on the same folder skips unchanged spans. `--reextract` forces re-extraction.

## 6. Query-time pipeline

`oragraphrag.retrieve.query(q: str)` runs:

**Step 0 — Embed query.** Same embedding model as ingest. Produces `q ∈ R^dim`.

**Step 1 — Seed retrieval (vector).** Top-K seeds via HNSW:

```sql
SELECT id, VECTOR_DISTANCE(embedding, :q, COSINE) AS d
FROM Entity ORDER BY d FETCH APPROX FIRST :k ROWS ONLY;
```

Defaults: K=8 entities, K=16 propositions.

**Step 2 — Bounded subgraph extraction (PGQL).** Two-hop neighborhood from the seed entities, capped at `MAX_NODES` (default 256):

```pgql
SELECT e1.id AS src, e2.id AS dst,
       r.predicate, r.ontology_axis, r.base_weight, r.support_propositions
FROM MATCH (e1:Entity) -[r:REL]-> (e2:Entity) ON oragraph
WHERE e1.id IN (:seed_ids) OR e2.id IN (:seed_ids)
FETCH FIRST :max_edges ROWS ONLY;
```

A second PGQL call expands one hop further only from the highest-degree seed nodes, to avoid combinatorial blowup on hubs.

**Step 3 — Edge reweighting (the novel bit, in-Python).** For each axis `a`:

```
proj[a] = cos(q, axis_embedding[a])               # in [-1, 1]
amp[a]  = sigmoid(α · proj[a] + β)                # in (0, 1); defaults α=8, β=0
```

Then for each edge `e` on axis `a`:

```
w(e) = base_weight(e) · amp[a]
```

Five scalars per query, O(|E_subgraph|), vectorized in NumPy. "Amplitude" is defined precisely as the sigmoid-squashed projection of the query onto each axis.

**Step 4 — Spreading activation.** Personalized PageRank with `igraph.Graph.personalized_pagerank(reset_vertices=seed_ids, weights="w", damping=0.85, implementation="prpack")`. Sub-1k-node subgraphs run in milliseconds.

**Step 5 — Proposition assembly.** Top-M activated entities (default M=20). Walk `support_propositions`, dedupe, rank by:

```
prop_score = max(activation(s), activation(o)) · prop_seed_similarity · confidence
```

Cap at a token budget (default 4k). Selection exposed as a JSON Duality View document containing `propositions`, `entities`, `edges_used`. `edges_used` powers the visualizer and the citations.

**Step 6 — Answer generation.** The JSON doc is rendered into `prompts/answer.j2` (system: strict grounding + cite proposition IDs; user: question). LLM (Grok 4.3 default) returns answer + citations; citations are mapped back to source docs via `Proposition.source_doc + source_span`.

**Latency targets** (Oracle docs corpus ~50k propositions):

- Step 1: ~15 ms
- Step 2: ~40 ms
- Step 3: <5 ms
- Step 4: ~20 ms
- Step 5: ~15 ms
- Step 6: 1–4 s (LLM dominates)
- End-to-end pre-LLM target: **<100 ms**

**Tunable knobs in `config.yaml`:**

- `seed_k_entities`, `seed_k_propositions`, `max_subgraph_nodes`, `max_subgraph_edges`
- `amplitude.alpha`, `amplitude.beta`, `amplitude.per_axis_overrides`
- `pagerank.damping`, `pagerank.top_m_entities`
- `answer.token_budget`

## 7. LLM integration and configuration

`oragraphrag.llm` exposes `complete(prompt, schema=None)` and `embed(texts)`.

```yaml
llm:
  provider: oci_grok        # oci_grok | ollama | openai_compat
  oci_grok:
    compartment_ocid: ...
    endpoint_id: ...
    model: grok-4-3
    region: us-chicago-1
  ollama:
    base_url: http://localhost:11434
    model: qwen3.5:35b-a3b
  fallback_on_outage: false   # if true, switches to ollama when oci_grok is unreachable
embeddings:
  provider: oracle          # oracle | ollama | sentence_transformers
  oracle:
    model: ALL_MINILM_L12_V2
  dim: 384
```

OCI auth follows the Oracle AI Developer Hub pattern: `OCI_CONFIG_FILE` env (default `~/.oci/config`) + profile, with instance-principal fallback inside OCI. Credentials never live in `config.yaml`.

Env-var overrides via `OGR__SECTION__KEY` for every value.

## 8. Prompts

Versioned under `src/oragraphrag/prompts/`:

- `extract.j2` — proposition + triple + ontology-axis extraction. Includes an axis rubric (one paragraph + 2–3 examples per axis) so axis tagging stays consistent across runs. Highest-leverage prompt.
- `answer.j2` — strict-grounding QA with required proposition-ID citations.
- `judge.j2` — LLM-as-judge for the benchmark, 0–4 rubric with citation check.

## 9. Benchmark harness

`oragraphrag.bench` runs `BenchSuite` JSONL files of `{question, gold_answer, gold_doc_ids}` against pluggable baseline runners.

v1 ships:

- `benchmarks/suites/oracle_docs_qa.jsonl` — 200 hand-curated Oracle 23ai docs questions: 80 single-hop, 80 two-hop, 40 three-hop. Gold answers + source spans frozen after one review pass.
- Baseline runners under `benchmarks/baselines/`:
  - `naive_rag` — 512-token fixed chunks, top-k vector retrieval on the same Oracle store, same LLM.
  - `graphrag` — Microsoft GraphRAG via their open package, same corpus + LLM.
  - `lightrag` — LightRAG dual-level retrieval, same corpus + LLM.
  - `oragraphrag` — this project.

All runners go through the same `LLM` adapter and embedding provider so the comparison isolates retrieval quality.

**Metrics** written to `benchmark_results.json` + `paper/tables/main.tex`:

- Answer correctness (Grok-as-judge, 0–4 rubric, median + bootstrap 95% CI).
- Citation precision/recall vs gold spans.
- Tokens-per-query (the Graphify-style efficiency number).
- Latency p50/p95 pre-LLM and end-to-end.
- Recall@k for multi-hop, bucketed by hop count.

`paper/figures/edge_amplitude_heatmap.pdf` shows one or two qualitative cases of which axes activate per question.

**Reproducibility.** All seeds fixed; baseline configs committed; LLM call cache in SQLite keyed by `(prompt_hash, model, params)`; `docker compose up && oragraphrag bench` reproduces from a clean machine.

## 10. Testing strategy

1. **Unit tests — pure compute.** `retrieve.py` is the largest target: hand-build a 30-node `igraph`, hard-code axis projections, assert PageRank rankings and edge reweightings to exact values. Runs in ~1 s.
2. **Schema-validation tests.** 20 golden extracted-proposition payloads (valid + invalid) exercise the JSON validator in `extract.py`.
3. **Prompt regression tests.** Opt-in (`--llm-tests`) fixture set for `extract`, `answer`, `judge`. Run before tagging releases and before paper submission. Catches axis-rubric drift — the #1 risk to the technique.
4. **Integration tests against Oracle 23ai Free.** Docker-compose stack; 5-doc corpus through `graphify`; 3 canned queries; asserts non-empty answers, valid citations, expected node/edge counts. Runs on a self-hosted runner.
5. **Benchmark smoke test.** `bench --suite oracle-docs-qa --limit 5 --systems oragraphrag,naive_rag` on every PR.
6. **Idempotency test.** `graphify` twice on the same folder; row counts unchanged; ledger reflects skipped spans.

Coverage targets: 85%+ for `retrieve`, `graph`, `extract`. LLM wrapper and CLI are integration-tested.

## 11. Error handling

- **LLM returns invalid JSON during extraction.** One retry with a "fix this JSON" prompt; on second failure, skip buffer, log to `logs/extract-failures.jsonl`, continue.
- **OCI GenAI rate-limited or unreachable.** Exponential backoff + jitter, max 5 attempts; fall back to Ollama if `llm.fallback_on_outage: true`.
- **Oracle DB connection lost mid-batch.** Transaction-per-buffer; in-flight buffer is retried; resume from the ledger.
- **Empty subgraph after seed retrieval.** Answer module returns "I don't have information about that in the indexed corpus" with no fabricated citations. Covered by an integration test.
- **All ontology amplitudes near zero.** Degenerate to `amp[a] = 1.0` for all axes (base-weight PageRank). Logged as a benchmark metric.
- **Embedding dim mismatch on startup.** Refuse to start; print "run `oragraphrag init-db --rebuild`".

Not handled: arbitrary malformed input documents (skipped with warning), manual edits to the property graph (undefined behavior), partial `init-db` failures (must rerun cleanly).

## 12. Rollout milestones

Each milestone is a working, demoable slice.

- **M1 — Skeleton + DB schema (1–2 days):** repo scaffold, `init-db` creates tables + HNSW + property graph, smoke insert+PGQL query.
- **M2 — Ingest pipeline (3–4 days):** `graphify` end-to-end on a synthetic 50-file corpus; LLM adapter (OCI Grok 4.3 + Ollama); extraction prompts + axis rubric.
- **M3 — Query pipeline (3 days):** seed → subgraph → reweighting → PageRank → JSON Duality View → answer; CLI `query` works.
- **M4 — Notebook + visualizer (1–2 days):** `oragraphrag_demo.ipynb` with `pyvis` graph plots and an amplitude heatmap.
- **M5 — Oracle docs corpus + benchmark suite (3–4 days):** real Oracle 23ai docs slice; 200-question suite; three baselines wired; bake-off; figures and tables generated.
- **M6 — Paper + polish (3 days):** NIPS-style `paper.tex` (via `jasperan-paper-writing-nips-style`), README, optional post-commit hub-sync.

Total: ~15–18 focused days. Each milestone produces a tagged artifact.

## 13. Out of scope for v1

- Web UI.
- Multi-ontology layered graphs (Approach C — paper's future work section).
- Conversation-aware spreading activation (paper's future work section). When this lands in v2, the working + episodic memory layer should be implemented via the [`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/) PyPI package (Oracle 23ai-backed agent memory with semantic/episodic/working tiers). Our property graph remains the semantic graph; `oracleagentmemory` provides the per-conversation thread context and `get_context_card()` that drives carry-forward amplitudes across turns.
- Sub-day incremental graph updates / online learning of ontology-axis vectors.
- Cross-corpus federation.

## 14. Open questions deferred to implementation

- Confirm `ALL_MINILM_L12_V2` (dim 384) is available via Oracle 23ai Free's built-in `VECTOR_EMBEDDING` at M1; if not, fall back to a `sentence_transformers` provider with the same model loaded locally.
- Exact Oracle docs slice for the corpus (likely the 23ai Concepts + SQL Reference + AI Vector Search guide subsections; locked during M5).
- Whether to publish OraGraphRAG as a PyPI package after v1; if yes, follow the PyPI workflow already established for `ragcli`.
