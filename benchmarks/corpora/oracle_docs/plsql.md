# PL/SQL APIs for AI Vector Search

The DBMS_VECTOR PL/SQL package exposes embedding and chunking utilities to SQL applications, removing the need to call external services for typical RAG ingest pipelines.

## DBMS_VECTOR

`DBMS_VECTOR` is the namespace for vector-related procedures and functions. Major surfaces: `UTL_TO_EMBEDDINGS` generates embeddings from input text, `UTL_TO_CHUNKS` splits long text into overlapping windows, and `CREATE_INDEX` builds a vector index programmatically. The package is granted via `GRANT EXECUTE ON DBMS_VECTOR TO <user>`.

## embedding

`DBMS_VECTOR.UTL_TO_EMBEDDINGS(text, model)` returns a VECTOR for the input text using the named embedding model. Bundled models include ALL_MINILM_L12_V2 (384 dim) and ALL_MINILM_L6_V2 (384 dim). Custom ONNX models can be loaded via `DBMS_VECTOR.LOAD_ONNX_MODEL` and referenced by name. The same model name appears in `VECTOR_EMBEDDING(:model USING :text AS data) FROM dual` for ad-hoc SQL invocations.

## utl_to_chunks

`DBMS_VECTOR.UTL_TO_CHUNKS(text, params)` splits text into chunks suitable for embedding. The `params` JSON controls chunk size (in characters or tokens), overlap, and split boundary preferences (sentence, paragraph, recursive). The function returns a collection that can be unnested in a SQL pipeline for direct insert into a VECTOR-bearing table.
