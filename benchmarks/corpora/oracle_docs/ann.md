# Approximate Nearest Neighbor Search in Oracle 23ai

Approximate nearest neighbor (ANN) search returns the K most similar vectors from a corpus without scanning every row. Oracle Database 23ai exposes ANN through the `VECTOR_DISTANCE` SQL function combined with a vector index.

## VECTOR_DISTANCE

`VECTOR_DISTANCE(v1, v2, metric)` returns the distance between two VECTOR values under the named metric. Supported metrics: COSINE, EUCLIDEAN, EUCLIDEAN_SQUARED, DOT, MANHATTAN, HAMMING. The function is index-aware: when one argument is a column with a vector index and the other is a bind variable, Oracle's optimizer uses the index to short-circuit exact distance computation.

A typical ANN query is:
```sql
SELECT id, VECTOR_DISTANCE(embedding, :q, COSINE) d
FROM proposition
ORDER BY d
FETCH APPROX FIRST 10 ROWS ONLY;
```

## approximate

The `FETCH APPROX FIRST` clause signals that approximate results are acceptable. Combined with an HNSW or IVF index, this triggers ANN traversal rather than a full table scan. Without `APPROX`, the query falls back to exact distance computation across all rows even when an index exists.

## exact

For small tables or when exact ranking matters, omit `APPROX` and let Oracle scan every row. Exact mode is also the right choice when validating an ANN index's recall during benchmarking. Exact-vs-approx parity is a useful regression check after index rebuilds.
