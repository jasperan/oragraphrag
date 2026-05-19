# /graphify skill — operator setup

This skill makes the OraGraphRAG ingest pipeline available as a Claude
Code slash command. It runs in the same environment Claude Code is
launched from, so the prerequisites are:

## Prerequisites

1. **OraGraphRAG installed**: `pip install -e .` from this repo, or
   pip install the published package.
2. **Conda env active**: `conda activate oragraphrag` (or whichever env
   has the CLI on PATH).
3. **Oracle 23ai Free container running**:
   ```
   docker compose up -d oracle-free
   ```
4. **LLM endpoint configured**: either OCI Grok 4.3 credentials (`oci`
   config + `OCI_COMPARTMENT_ID`) or a local Ollama with `gemma3:4b` or
   larger pulled.
5. **Schema initialized once**: `oragraphrag init-db --rebuild`.

## Per-project config

By default the skill looks for `./config.yaml` in the folder being
graphified. Override with the `OGR_CONFIG` env var pointing at an
absolute path:

```bash
export OGR_CONFIG=/path/to/my/oragraphrag.yaml
```

## Invocation

In any Claude Code session, `cd` into the folder you want to ingest and
type:

```
/graphify
```

The skill blocks until ingest completes (a few seconds for small
folders, several minutes for large ones) and prints the JSON stats +
the assigned source_id.

## Scoping queries to a specific source

After graphifying multiple folders, you can scope a query to one
ingest:

```bash
oragraphrag query "What did we learn about HNSW?" --source /path/to/folder
```

This filters retrieval to only the propositions + edges tagged with
that source_id.

## Removing a skill

Skills are file-system-resolved. Delete `.claude/skills/graphify/` to
disable.
