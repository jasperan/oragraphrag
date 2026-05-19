# AI Vector Search

AI Vector Search is the umbrella name for the vector-database capabilities Oracle Database 23ai ships: VECTOR datatype, vector indexes, distance functions, and DBMS_VECTOR PL/SQL APIs. The pipeline supports RAG, semantic search, recommendation, and anomaly detection workloads without requiring an external vector store.

## pipeline

A typical AI Vector Search pipeline is: ingest raw documents → chunk via `DBMS_VECTOR.UTL_TO_CHUNKS` → embed via `DBMS_VECTOR.UTL_TO_EMBEDDINGS` → store in a VECTOR column → create an HNSW or IVF index → query with `VECTOR_DISTANCE` and `FETCH APPROX FIRST`. Every step runs inside the database; no external service is needed.

## rag

Retrieval-Augmented Generation (RAG) is the canonical use case. The user's question is embedded, the database returns the K nearest chunks, and an external LLM (or one called from PL/SQL via DBMS_VECTOR's LLM integration) generates an answer grounded in the retrieved chunks. Oracle's AI Vector Search supplies the retrieval half of the pipeline with ACID guarantees the application would otherwise need to layer on top of a separate vector store.

## tokens

For RAG, chunk size in tokens matters because the retrieved context must fit the answering LLM's context window. The recommended chunk sizes are 256–512 tokens with 50–100 token overlap. `UTL_TO_CHUNKS` accepts a `max` parameter expressed in tokens (using the chosen tokenizer) for predictable budget enforcement.
