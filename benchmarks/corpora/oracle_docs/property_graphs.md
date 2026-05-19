# Property Graphs in Oracle Database 23ai

Oracle Database 23ai introduced native SQL property graphs queryable with PGQL via the `GRAPH_TABLE` SQL operator. Property graphs are defined as views over existing relational tables, so the same data can be queried with SQL, PL/SQL, and PGQL without duplication.

## pgql

PGQL (Property Graph Query Language) expresses graph pattern matches: `MATCH (a:Entity) -[r:REL]-> (b:Entity)` finds edges of label REL between Entity vertices. Predicates, projections, and aggregations follow SQL conventions. In Oracle 23ai, a PGQL query is wrapped in `GRAPH_TABLE(graph_name MATCH ... COLUMNS (...))` and consumed like any subquery.

## create

`CREATE PROPERTY GRAPH name VERTEX TABLES (...) EDGE TABLES (...)` defines a graph over existing tables. Each vertex table contributes a node label and a key column; each edge table contributes an edge label, a source key, a destination key, and zero or more edge properties. The graph view is logical — no data is copied.

## vertex_table

Vertex tables enumerate the entities in the graph. A typical declaration is `Entity KEY (id) LABEL Entity PROPERTIES (id, name, kind)`. The KEY clause names the unique column used as the vertex identifier; PROPERTIES lists the columns exposed to PGQL. Multiple vertex tables can coexist in one graph, each with its own label, to model heterogeneous node types.
