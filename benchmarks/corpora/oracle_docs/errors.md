# Common AI Vector Search Errors

Oracle Database 23ai's AI Vector Search and property graph features surface three frequent error codes that operators encounter during initial setup. Each has a deterministic root cause and a single fix.

## ORA-21560

`ORA-21560: argument N is null, invalid, or out of range`. Most commonly seen when a NULL embedding is passed to `VECTOR_DISTANCE` or to an HNSW index probe. The fix is to ensure the embedding column is NOT NULL or to filter NULL rows from the candidate set before the distance computation. NULL embeddings typically indicate an upstream ingest bug where the embedder backend returned an empty or sentinel value for an empty-string input.

## ORA-43853

`ORA-43853: VECTOR type cannot be used in non-automatic segment space management tablespace "SYSTEM"`. The default SYSTEM tablespace in FREEPDB1 uses manual segment space management, which the VECTOR storage layout doesn't support. Fix: create a USERS tablespace with `SEGMENT SPACE MANAGEMENT AUTO` and assign the user to it via `ALTER USER ... DEFAULT TABLESPACE USERS`.

## ORA-51962

`ORA-51962: The vector memory area is out of space for the current container`. HNSW indexes reside in a dedicated vector memory pool that defaults to 0 in the Oracle 23ai Free `:latest-lite` container image. Fix: `ALTER SYSTEM SET vector_memory_size=512M SCOPE=SPFILE` (or larger), then restart the database. The setting cannot be changed without an instance restart.
