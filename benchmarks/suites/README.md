# Bench suites

## oracle_docs_qa.jsonl

200 questions probing Oracle 23ai documentation across single-hop, two-hop,
and three-hop reasoning.

**STATUS: SCAFFOLDING, NOT YET HUMAN-VALIDATED.**

The current content was generated programmatically from topic templates to
let the bench harness exercise its full pipeline shape (80/80/40 split by
hops, plausible `doc#section` references, real Oracle terminology). Before
the paper benchmark is run, a human curator must:

1. Replace placeholder gold answers with answers grounded in the actual
   Oracle 23ai documentation slice being indexed.
2. Verify that the `gold_doc_ids` point at sections that actually exist
   in the indexed corpus.
3. Refine multi-hop questions so they genuinely require the listed hop
   count (the templates produce plausible but not always strictly
   multi-hop questions).
4. Add a small number of negative-control questions (questions whose
   answers are *not* in the corpus) to exercise the "I don't have
   information" path.

The 5-question smoke suite from Task 15 (now `oracle_docs_qa.smoke.jsonl`)
remains available for fast iteration.

## Distribution

| Hops | Count | IDs           |
|------|------:|---------------|
| 1    |    80 | q001 .. q080  |
| 2    |    80 | q081 .. q160  |
| 3    |    40 | q161 .. q200  |

## Topical coverage

The scaffolding draws from eight Oracle 23ai topical clusters:

- `vectors` — VECTOR datatype, format, precision
- `indexes` — HNSW, IVF, accuracy tuning
- `ann` — VECTOR_DISTANCE, approximate vs. exact search
- `plsql` — DBMS_VECTOR, embedding APIs, UTL_TO_CHUNKS
- `ai_vector_search` — pipeline, RAG, token accounting
- `json_duality` — views, schema, validation
- `property_graphs` — PGQL, CREATE PROPERTY GRAPH, vertex tables
- `errors` — ORA-21560, ORA-43853, ORA-51962
