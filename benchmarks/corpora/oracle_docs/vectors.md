# Vectors in Oracle Database 23ai

Oracle Database 23ai introduced first-class support for vector embeddings as a native SQL datatype, enabling semantic similarity search alongside relational queries without an external vector store.

## datatype

The VECTOR datatype stores fixed-length numeric vectors. A column is declared with both a dimension count and an element format, for example `VECTOR(384, FLOAT32)`. Supported formats are FLOAT32, FLOAT64, INT8, and BINARY. The dimension is required at table-creation time and cannot be changed without recreating the column.

A VECTOR column participates in standard SQL operations: INSERT, UPDATE, DELETE, and SELECT all work as expected. Vectors can also be NULL.

## format

The format determines per-element storage size and precision. FLOAT32 (4 bytes per element) is the default for most modern embedding models and balances storage with quality. FLOAT64 doubles storage for diminishing returns on typical embeddings. INT8 enables quantized vectors at one-quarter the size of FLOAT32 with minor accuracy loss. BINARY packs vectors at 1 bit per element for extreme compression.

## precision

Precision tradeoffs matter for AI Vector Search workloads. FLOAT32 is recommended for most ANN scenarios where the embedding model itself produces 32-bit weights. Quantized formats (INT8, BINARY) reduce storage and improve cache locality at the cost of recall in similarity search. The VECTOR datatype does not enforce normalization — the application is responsible for L2-normalizing vectors if the chosen distance metric assumes unit vectors.
