---
name: graphify
description: Ingest the current folder into the OraGraphRAG Oracle 23ai property graph. Triggers when the user types /graphify or asks to graphify, ingest, or index a folder/codebase into the knowledge graph.
---

# /graphify — ingest the current folder into OraGraphRAG

You are running the OraGraphRAG ingest pipeline against the current
working directory. Each run produces a namespaced subgraph keyed on the
folder path; re-running on the same folder is idempotent (existing
buffers are skipped via the IngestLedger).

## Pre-flight checks

Before running, verify the environment:

1. `oragraphrag --version` works (the CLI is installed and on PATH).
2. Either `./config.yaml` exists in the folder being graphified OR a
   global config is set via the `OGR_CONFIG` env var. If neither, abort
   and tell the user.
3. Oracle 23ai container is healthy:
   `docker exec oragraphrag-oracle /opt/oracle/checkDBStatus.sh` returns 0.
4. Either Ollama is up OR OCI Grok 4.3 credentials are configured per
   the active config.yaml.

## Execution

Run synchronously, blocking until done:

```bash
oragraphrag graphify "$(pwd)" --config "${OGR_CONFIG:-./config.yaml}"
```

Capture the JSON stats output. It looks like:

```json
{
  "buffers": 32,
  "skipped": 0,
  "failed": 1,
  "propositions": 127,
  "rels": 133,
  "entities": 232,
  "source_id": "src_<hex>"
}
```

## Report back

Tell the user, in one short paragraph:

- The `source_id` that was assigned (so they can refer to it for scoped queries).
- How many propositions and edges were ingested.
- How many buffers failed (and recommend checking `logs/extract-failures.jsonl` if non-zero).
- That subsequent queries can scope to this source via `oragraphrag query "..." --source $(pwd)`.

If graphify failed entirely (non-zero exit), surface the error message and
do not pretend it succeeded.

## Why this matters

Each `/graphify` builds a per-source subgraph in the shared Oracle 23ai
instance. Future Claude Code sessions can query across all ingested
sources, enabling continual learning: knowledge from session N is
available in session N+M without re-ingesting.
