# Oracle 23ai Vector Concepts

The VECTOR datatype stores fixed-length numeric vectors and powers AI Vector Search.
Each vector has a declared dimension and element format (FLOAT32 by default).

## Indexes

HNSW is an approximate nearest-neighbor index for VECTOR columns.
It uses a hierarchical small-world graph to accelerate similarity search.
The COSINE distance metric is supported alongside EUCLIDEAN and DOT.

## Property Graphs

Oracle Database 23ai introduced native SQL property graphs queryable with GRAPH_TABLE.
You define vertex and edge tables and CREATE PROPERTY GRAPH composes them.

## PL/SQL APIs

The DBMS_VECTOR package exposes embedding and chunking utilities.
UTL_TO_EMBEDDINGS generates embeddings from text.
UTL_TO_CHUNKS splits long text into overlapping windows.
