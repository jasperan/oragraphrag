"""Oracle 23ai property-graph + vector store. The only module that talks to the DB."""

from __future__ import annotations

import array
import contextlib
import json
from collections.abc import Iterable
from importlib.resources import files

import oracledb

from oragraphrag.config import Config


def _vec(values: list[float]) -> array.array:
    """Encode a Python list as the FLOAT32 array oracledb needs for VECTOR binds."""
    return array.array("f", values)


class GraphStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pool: oracledb.ConnectionPool | None = None

    def connect(self) -> None:
        if self._pool is None:
            self._pool = oracledb.create_pool(
                user=self.cfg.oracle.username,
                password=self.cfg.oracle.password,
                dsn=self.cfg.oracle.dsn,
                min=self.cfg.oracle.pool_min,
                max=self.cfg.oracle.pool_max,
                increment=1,
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _conn(self):
        assert self._pool is not None, "call connect() first"
        return self._pool.acquire()

    # ---------- DDL ----------

    def init_db(self, *, rebuild: bool, axis_vectors: dict[str, list[float]]) -> None:
        dim = self.cfg.embeddings.dim
        schema_sql = (
            files("oragraphrag.sql")
            .joinpath("schema.sql")
            .read_text()
            .replace(":dim", str(dim))
        )
        with self._conn() as c, c.cursor() as cur:
            if rebuild:
                for stmt in (
                    "DROP PROPERTY GRAPH oragraph",
                    "DROP TABLE Mentions PURGE",
                    "DROP TABLE Rel PURGE",
                    "DROP TABLE Proposition PURGE",
                    "DROP TABLE Entity PURGE",
                    "DROP TABLE Ontology_Axis PURGE",
                    "DROP TABLE Ingest_Ledger PURGE",
                ):
                    with contextlib.suppress(oracledb.DatabaseError):
                        cur.execute(stmt)
            for stmt in self._split_sql(schema_sql):
                cur.execute(stmt)
            for name, vec in axis_vectors.items():
                cur.execute(
                    "INSERT INTO Ontology_Axis (name, description, axis_embedding) "
                    "VALUES (:n, :d, :v)",
                    n=name,
                    d=name,
                    v=_vec(vec),
                )
            c.commit()

    @staticmethod
    def _split_sql(text: str) -> list[str]:
        """Split a multi-statement SQL script on `;`.

        Treats `CREATE PROPERTY GRAPH ... ;` as a single statement so the trailing
        semicolon inside the body (after the closing paren) is the terminator.
        """
        out: list[str] = []
        buf: list[str] = []
        in_block = False
        for raw_line in text.splitlines():
            line = raw_line
            stripped_lower = line.strip().lower()
            # Skip blank lines and SQL line comments outside of any statement
            if not buf and (not stripped_lower or stripped_lower.startswith("--")):
                continue
            if not in_block and stripped_lower.startswith("create property graph"):
                in_block = True
            buf.append(line)
            stripped = line.rstrip()
            if stripped.endswith(";"):
                stmt = "\n".join(buf).rstrip()
                # Drop trailing semicolon — cx_Oracle/oracledb doesn't want it.
                if stmt.endswith(";"):
                    stmt = stmt[:-1].rstrip()
                if stmt:
                    out.append(stmt)
                buf = []
                in_block = False
        tail = "\n".join(buf).strip()
        if tail:
            out.append(tail.rstrip(";").rstrip())
        return [s for s in out if s]

    def list_tables(self) -> list[str]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT table_name FROM user_tables")
            return [r[0] for r in cur.fetchall()]

    def list_property_graphs(self) -> list[str]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT graph_name FROM user_property_graphs")
            return [r[0] for r in cur.fetchall()]

    # ---------- Entity ops ----------

    def upsert_entity(self, *, name: str, kind: str, embedding: list[float]) -> bytes:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT id FROM Entity WHERE LOWER(name) = LOWER(:n)", n=name)
            row = cur.fetchone()
            if row:
                eid = row[0]
                cur.execute(
                    "UPDATE Entity SET mention_count = mention_count + 1, "
                    "embedding = :v WHERE id = :id",
                    v=_vec(embedding),
                    id=eid,
                )
            else:
                out_id = cur.var(oracledb.DB_TYPE_RAW)
                cur.execute(
                    "INSERT INTO Entity (name, kind, embedding) "
                    "VALUES (:n, :k, :v) RETURNING id INTO :id",
                    n=name,
                    k=kind,
                    v=_vec(embedding),
                    id=out_id,
                )
                eid = out_id.getvalue()[0]
            c.commit()
            return eid

    def upsert_proposition(
        self, *, text: str, source_doc: str, source_span: str, embedding: list[float]
    ) -> bytes:
        with self._conn() as c, c.cursor() as cur:
            out_id = cur.var(oracledb.DB_TYPE_RAW)
            cur.execute(
                "INSERT INTO Proposition (text, source_doc, source_span, embedding) "
                "VALUES (:t, :d, :s, :v) RETURNING id INTO :id",
                t=text,
                d=source_doc,
                s=source_span,
                v=_vec(embedding),
                id=out_id,
            )
            c.commit()
            return out_id.getvalue()[0]

    def upsert_rel(
        self,
        src_id: bytes,
        dst_id: bytes,
        *,
        predicate: str,
        ontology_axis: str,
        base_weight: float,
        support_prop_id: bytes,
    ) -> None:
        """Insert or update a REL row keyed on (src_id, dst_id, predicate).

        Simplified vs. the plan's literal MERGE: we do a SELECT-then-INSERT/UPDATE
        in Python so we can mutate the JSON arrays (support_propositions,
        support_axis_counts) without depending on JSON_ARRAY_APPEND / JSON_TABLE
        functions that are not uniformly available across 23ai builds. The
        most-frequent-axis resolution logic is implemented here in Python so
        it lives in one place and is testable. Behavior matches the spec §5 step 5:
        if axes disagree, the most-frequent axis wins; ties broken by most-recent.
        """
        sp_hex = support_prop_id.hex()
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT id, base_weight, support_propositions, support_axis_counts, "
                "ontology_axis FROM Rel "
                "WHERE src_id = :s AND dst_id = :d AND predicate = :p",
                s=src_id,
                d=dst_id,
                p=predicate,
            )
            row = cur.fetchone()
            if row is None:
                support_props = [sp_hex]
                axis_counts = {ontology_axis: 1}
                cur.execute(
                    "INSERT INTO Rel (src_id, dst_id, predicate, ontology_axis, "
                    "base_weight, support_propositions, support_axis_counts) "
                    "VALUES (:s, :d, :p, :a, :bw, :sp, :ac)",
                    s=src_id,
                    d=dst_id,
                    p=predicate,
                    a=ontology_axis,
                    bw=base_weight,
                    sp=json.dumps(support_props),
                    ac=json.dumps(axis_counts),
                )
            else:
                rid, old_bw, old_sp, old_ac, _old_axis = row
                support_props = self._read_json(old_sp) or []
                axis_counts = self._read_json(old_ac) or {}
                support_props.append(sp_hex)
                axis_counts[ontology_axis] = int(axis_counts.get(ontology_axis, 0)) + 1
                # Most-frequent axis wins. The current call's axis is the most-recent
                # assertion, so it acts as the tie-breaker (we iterate axis_counts in
                # insertion order but always promote the just-seen axis on tie).
                best_axis = ontology_axis
                best_count = axis_counts[ontology_axis]
                for k, v in axis_counts.items():
                    if v > best_count:
                        best_axis = k
                        best_count = v
                new_bw = (float(old_bw) + base_weight) / 2.0
                cur.execute(
                    "UPDATE Rel SET base_weight = :bw, last_seen_at = SYSTIMESTAMP, "
                    "support_propositions = :sp, support_axis_counts = :ac, "
                    "ontology_axis = :a WHERE id = :id",
                    bw=new_bw,
                    sp=json.dumps(support_props),
                    ac=json.dumps(axis_counts),
                    a=best_axis,
                    id=rid,
                )
            c.commit()

    @staticmethod
    def _read_json(value: object) -> object:
        """Decode an Oracle JSON column value into a Python object.

        Oracle's JSON type can come back as a Python dict/list directly (native
        JSON), a str, or a LOB depending on the driver mode. Be liberal.
        """
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, (bytes, bytearray)):
            return json.loads(bytes(value).decode("utf-8"))
        if hasattr(value, "read"):
            return json.loads(value.read())
        if isinstance(value, str):
            return json.loads(value)
        return value

    # ---------- Reads ----------

    def vector_search_entities(self, *, query_vec: list[float], k: int) -> list[dict]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                FROM Entity ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                """,
                q=_vec(query_vec),
                k=k,
            )
            return [
                {"id": r[0], "name": r[1], "distance": float(r[2])} for r in cur.fetchall()
            ]

    def vector_search_propositions(self, *, query_vec: list[float], k: int) -> list[dict]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_doc, source_span,
                       VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                FROM Proposition ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                """,
                q=_vec(query_vec),
                k=k,
            )
            return [
                {
                    "id": r[0],
                    "source_doc": r[1],
                    "source_span": r[2],
                    "distance": float(r[3]),
                }
                for r in cur.fetchall()
            ]

    def pgql_subgraph(self, *, seed_ids: list[bytes], max_edges: int) -> list[dict]:
        """Return REL edges within one hop of any seed (i.e. seeds as either endpoint).

        Implementation notes:
        - VERTEX_ID() in 23ai returns a JSON composite key, not the raw RAW(16),
          which makes IN-list filtering against a bytes bind never match. We
          project `e.id` (the underlying property) instead — that comes back as
          plain bytes.
        - The :max_edges placeholder must be padded out as :m and bound separately
          because of the bind-parameter ordering rules around `FETCH FIRST`.
        """
        if not seed_ids:
            return []
        placeholders = ", ".join(f":s{i}" for i in range(len(seed_ids)))
        bind = {f"s{i}": sid for i, sid in enumerate(seed_ids)}
        bind["m"] = max_edges
        sql = f"""
            SELECT e1_id, e2_id, predicate, ontology_axis, base_weight,
                   support_propositions
            FROM GRAPH_TABLE (oragraph
              MATCH (e1) -[r IS REL]-> (e2)
              WHERE e1.id IN ({placeholders})
                 OR e2.id IN ({placeholders})
              COLUMNS (e1.id AS e1_id, e2.id AS e2_id,
                       r.predicate AS predicate,
                       r.ontology_axis AS ontology_axis,
                       r.base_weight AS base_weight,
                       r.support_propositions AS support_propositions)
            )
            FETCH FIRST :m ROWS ONLY
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, **bind)
            rows = cur.fetchall()
        return [
            {
                "src": bytes(r[0]) if r[0] is not None else None,
                "dst": bytes(r[1]) if r[1] is not None else None,
                "predicate": r[2],
                "ontology_axis": r[3],
                "base_weight": float(r[4]),
                "support_propositions": self._read_json(r[5]) or [],
            }
            for r in rows
        ]

    def fetch_propositions(self, ids: Iterable[bytes]) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ", ".join(f":i{i}" for i in range(len(ids)))
        bind = {f"i{i}": v for i, v in enumerate(ids)}
        sql = (
            "SELECT id, text, source_doc, source_span FROM Proposition "
            f"WHERE id IN ({placeholders})"
        )
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, **bind)
            return [
                {
                    "id": r[0],
                    "text": r[1].read() if hasattr(r[1], "read") else (r[1] or ""),
                    "source_doc": r[2],
                    "source_span": r[3],
                }
                for r in cur.fetchall()
            ]


