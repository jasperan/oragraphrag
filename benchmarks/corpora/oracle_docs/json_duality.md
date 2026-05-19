# JSON Relational Duality Views

JSON Relational Duality Views (introduced in Oracle Database 23ai) expose normalized relational tables as JSON documents and vice versa. A single underlying table set serves both SQL queries and document-oriented APIs without data duplication.

## views

A duality view is declared with `CREATE JSON RELATIONAL DUALITY VIEW`. The view's body describes how relational rows map to JSON structure: parent/child relationships, embedded vs referenced shapes, and ETag-based optimistic locking semantics. Updates through the duality view propagate back to the underlying tables with full ACID guarantees.

## schema

The duality view's JSON schema is derived from the underlying SQL schema plus the mapping clauses in the view definition. Nested objects map to child tables joined by foreign keys; arrays map to one-to-many relations; primitive fields map to columns. `DBMS_JSON_SCHEMA` can extract the generated JSON schema for client validation.

## validation

Duality views support JSON Schema validation. The `JSON_SCHEMA_VALID` function and `IS JSON VALIDATE USING` constraint enforce that incoming JSON documents conform to the declared schema before the update is committed to the underlying tables. This combines document-store ergonomics with relational integrity in one transaction.
