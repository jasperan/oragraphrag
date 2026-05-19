# Vector Indexes in Oracle Database 23ai

Vector indexes accelerate similarity search over VECTOR columns. Oracle 23ai supports two index families: HNSW (Hierarchical Navigable Small World) and IVF (Inverted File). Both are approximate nearest-neighbor indexes optimized for large-scale ANN search.

## HNSW

HNSW is a graph-based approximate nearest-neighbor index. It builds a hierarchical navigable small-world graph over the indexed vectors, layering shortcut edges across the dataset for logarithmic traversal time. HNSW indexes are created with `ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95` and consume vector memory pool space, which must be enabled via `ALTER SYSTEM SET vector_memory_size=512M SCOPE=SPFILE` before the database restart.

HNSW excels at high recall under low latency. It is the recommended default for AI Vector Search.

## IVF

IVF (Inverted File) clusters the dataset into Voronoi cells at index-build time. Query-time search visits only a small subset of cells, trading recall for speed. IVF indexes are persisted to disk (unlike HNSW's in-memory residency) and rebuild faster on bulk inserts. IVF is the right choice when the corpus is too large to fit in the vector memory pool.

## accuracy

Both index families expose an accuracy knob. For HNSW, `WITH TARGET ACCURACY 95` instructs Oracle to tune the graph parameters (M, efConstruction) for ~95% recall against an exhaustive search baseline. For IVF, the number of probed cells per query controls the recall-vs-latency tradeoff. Higher target accuracy increases index-build cost and per-query work.
