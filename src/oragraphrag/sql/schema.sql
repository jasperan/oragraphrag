-- Run as the ORAGRAPH schema owner. Idempotent against a fresh schema; use
-- DROP+CREATE only in --rebuild mode (handled in Python before running this).

CREATE TABLE Entity (
  id         RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
  name       VARCHAR2(512) NOT NULL,
  kind       VARCHAR2(64),
  embedding  VECTOR(:dim, FLOAT32),
  mention_count NUMBER DEFAULT 0,
  created_at TIMESTAMP DEFAULT SYSTIMESTAMP
);
CREATE UNIQUE INDEX entity_name_uq ON Entity (LOWER(name));

CREATE TABLE Proposition (
  id           RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
  text         CLOB NOT NULL,
  source_doc   VARCHAR2(512),
  source_span  VARCHAR2(64),
  embedding    VECTOR(:dim, FLOAT32),
  created_at   TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE TABLE Rel (
  id                   RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
  src_id               RAW(16) NOT NULL REFERENCES Entity(id),
  dst_id               RAW(16) NOT NULL REFERENCES Entity(id),
  predicate            VARCHAR2(128) NOT NULL,
  ontology_axis        VARCHAR2(32) NOT NULL,
  base_weight          NUMBER NOT NULL,
  support_propositions JSON,
  support_axis_counts  JSON,
  created_at           TIMESTAMP DEFAULT SYSTIMESTAMP,
  last_seen_at         TIMESTAMP DEFAULT SYSTIMESTAMP,
  CONSTRAINT rel_triple_uq UNIQUE (src_id, dst_id, predicate)
);
CREATE INDEX rel_axis_ix ON Rel (ontology_axis);
CREATE INDEX rel_pred_ix ON Rel (predicate);

CREATE TABLE Mentions (
  proposition_id RAW(16) NOT NULL REFERENCES Proposition(id),
  entity_id      RAW(16) NOT NULL REFERENCES Entity(id),
  role           VARCHAR2(16) NOT NULL CHECK (role IN ('subject','object')),
  confidence     NUMBER,
  PRIMARY KEY (proposition_id, entity_id, role)
);

CREATE TABLE Ontology_Axis (
  name           VARCHAR2(32) PRIMARY KEY,
  description    VARCHAR2(2048),
  axis_embedding VECTOR(:dim, FLOAT32)
);

CREATE TABLE Ingest_Ledger (
  span_hash    VARCHAR2(64) PRIMARY KEY,
  doc_id       VARCHAR2(512),
  section_path VARCHAR2(1024),
  ingested_at  TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE VECTOR INDEX entity_emb_hnsw_ix ON Entity (embedding)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH
  DISTANCE COSINE WITH TARGET ACCURACY 95;

CREATE VECTOR INDEX prop_emb_hnsw_ix ON Proposition (embedding)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH
  DISTANCE COSINE WITH TARGET ACCURACY 95;

CREATE PROPERTY GRAPH oragraph
  VERTEX TABLES (
    Entity      KEY (id) LABEL Entity PROPERTIES (id, name, kind),
    Proposition KEY (id) LABEL Proposition PROPERTIES (id, text, source_doc, source_span)
  )
  EDGE TABLES (
    Rel
      KEY (id)
      SOURCE      KEY (src_id) REFERENCES Entity (id)
      DESTINATION KEY (dst_id) REFERENCES Entity (id)
      LABEL REL PROPERTIES (predicate, ontology_axis, base_weight, support_propositions),
    Mentions
      KEY (proposition_id, entity_id, role)
      SOURCE      KEY (proposition_id) REFERENCES Proposition (id)
      DESTINATION KEY (entity_id) REFERENCES Entity (id)
      LABEL MENTIONS PROPERTIES (role, confidence)
  );
