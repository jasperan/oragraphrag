"""Oracle 23ai property-graph + vector store. The only module that talks to the DB."""

from __future__ import annotations

import array
import json
from collections.abc import Iterable, Iterator
from importlib.resources import files

import oracledb
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from oragraphrag.config import Config
from oragraphrag.ingest_records import IngestUnit, IngestWriteStats

# Cap support_propositions to avoid unbounded JSON growth per edge under
# repeated assertions of the same triple. Keep the most-recent N hex ids.
_MAX_SUPPORT_PROPS = 100

# Oracle error codes that mean "drop target is missing" and are safe to
# suppress during init_db(rebuild=True). Anything else must surface.
_DROP_OK_ERROR_CODES = {942, 1418, 4043, 42421, 65042}
# ORA-00942: table or view does not exist
# ORA-01418: specified index does not exist
# ORA-04043: object does not exist
# ORA-42421: property graph does not exist
# ORA-65042: graph does not exist (alt code seen in some 23ai releases)


class GraphStoreError(RuntimeError):
    """Store-level error raised when a graph write unit cannot be completed."""


def _is_drop_target_missing(e: oracledb.DatabaseError) -> bool:
    """True iff a DROP DDL failed because the target object does not exist."""
    try:
        return e.args[0].code in _DROP_OK_ERROR_CODES
    except (AttributeError, IndexError):
        return False


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
            try:
                if rebuild:
                    # Indexes are dropped explicitly ahead of their tables so a
                    # stale VECTOR$..._HNSW_IX$ artifact from a half-finished
                    # previous run cannot block the rebuild.
                    drop_stmts = [
                        "DROP INDEX entity_emb_hnsw_ix",
                        "DROP INDEX prop_emb_hnsw_ix",
                        "DROP INDEX entity_name_uq",
                        "DROP INDEX entity_source_ix",
                        "DROP INDEX prop_source_ix",
                        "DROP INDEX rel_source_ix",
                        "DROP PROPERTY GRAPH oragraph",
                        "DROP TABLE Mentions PURGE",
                        "DROP TABLE Rel PURGE",
                        "DROP TABLE Proposition PURGE",
                        "DROP TABLE Entity PURGE",
                        "DROP TABLE Ontology_Axis PURGE",
                        "DROP TABLE Ingest_Ledger PURGE",
                    ]
                    for stmt in drop_stmts:
                        try:
                            cur.execute(stmt)
                        except oracledb.DatabaseError as e:
                            if not _is_drop_target_missing(e):
                                raise
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
            except Exception:
                c.rollback()
                raise

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

    def upsert_entity(
        self,
        *,
        name: str,
        kind: str,
        embedding: list[float],
        source_id: str = "default",
    ) -> bytes:
        with self._conn() as c, c.cursor() as cur:
            try:
                eid = self._upsert_entity(
                    cur,
                    name=name,
                    kind=kind,
                    embedding=embedding,
                    source_id=source_id,
                )
                c.commit()
                return eid
            except Exception:
                c.rollback()
                raise

    def _upsert_entity(
        self,
        cur,
        *,
        name: str,
        kind: str,
        embedding: list[float],
        source_id: str,
    ) -> bytes:
        cur.execute("SELECT id FROM Entity WHERE LOWER(name) = LOWER(:n)", n=name)
        row = cur.fetchone()
        if row:
            eid = row[0]
            cur.execute(
                "UPDATE Entity SET mention_count = mention_count + 1, "
                "embedding = :v, source_id = :sid WHERE id = :id",
                v=_vec(embedding),
                sid=source_id,
                id=eid,
            )
            return eid

        out_id = cur.var(oracledb.DB_TYPE_RAW)
        cur.execute(
            "INSERT INTO Entity (name, kind, embedding, source_id) "
            "VALUES (:n, :k, :v, :sid) RETURNING id INTO :id",
            n=name,
            k=kind,
            v=_vec(embedding),
            sid=source_id,
            id=out_id,
        )
        return out_id.getvalue()[0]

    def upsert_proposition(
        self,
        *,
        text: str,
        source_doc: str,
        source_span: str,
        embedding: list[float],
        source_id: str = "default",
    ) -> bytes:
        with self._conn() as c, c.cursor() as cur:
            try:
                pid = self._upsert_proposition(
                    cur,
                    text=text,
                    source_doc=source_doc,
                    source_span=source_span,
                    embedding=embedding,
                    source_id=source_id,
                )
                c.commit()
                return pid
            except Exception:
                c.rollback()
                raise

    def _upsert_proposition(
        self,
        cur,
        *,
        text: str,
        source_doc: str,
        source_span: str,
        embedding: list[float],
        source_id: str,
    ) -> bytes:
        out_id = cur.var(oracledb.DB_TYPE_RAW)
        cur.execute(
            "INSERT INTO Proposition (text, source_doc, source_span, embedding, source_id) "
            "VALUES (:t, :d, :s, :v, :sid) RETURNING id INTO :id",
            t=text,
            d=source_doc,
            s=source_span,
            v=_vec(embedding),
            sid=source_id,
            id=out_id,
        )
        return out_id.getvalue()[0]

    @retry(
        retry=retry_if_exception_type(oracledb.IntegrityError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.05, max=0.5),
        reraise=True,
    )
    def upsert_rel(
        self,
        src_id: bytes,
        dst_id: bytes,
        *,
        predicate: str,
        ontology_axis: str,
        base_weight: float,
        support_prop_id: bytes,
        source_id: str = "default",
    ) -> None:
        """Insert or update a REL row keyed on (src_id, dst_id, predicate)."""
        with self._conn() as c, c.cursor() as cur:
            try:
                self._upsert_rel(
                    cur,
                    src_id,
                    dst_id,
                    predicate=predicate,
                    ontology_axis=ontology_axis,
                    base_weight=base_weight,
                    support_prop_id=support_prop_id,
                    source_id=source_id,
                )
                c.commit()
            except Exception:
                c.rollback()
                raise

    def _upsert_rel(
        self,
        cur,
        src_id: bytes,
        dst_id: bytes,
        *,
        predicate: str,
        ontology_axis: str,
        base_weight: float,
        support_prop_id: bytes,
        source_id: str,
    ) -> None:
        """Insert or update a REL row inside the caller's transaction.

        Concurrency model: SELECT ... FOR UPDATE serializes concurrent writers
        on the same (src, dst, predicate) triple inside the transaction. The
        unique constraint rel_triple_uq gates the no-row INSERT race; on
        IntegrityError (another writer inserted between our SELECT-no-row and
        our INSERT) the @retry decorator restarts the whole method from a
        fresh SELECT, where the row will now be found and updated instead.

        Simplified vs. the plan's literal MERGE: we do a SELECT-then-INSERT/UPDATE
        in Python so we can mutate the JSON arrays (support_propositions,
        support_axis_counts) without depending on JSON_ARRAY_APPEND / JSON_TABLE
        functions that are not uniformly available across 23ai builds. The
        most-frequent-axis resolution logic is implemented here in Python so
        it lives in one place and is testable. Behavior matches the spec §5 step 5:
        if axes disagree, the most-frequent axis wins; ties broken by most-recent.
        """
        sp_hex = support_prop_id.hex()
        # FOR UPDATE serializes concurrent writers on the same triple.
        # On a fresh row, the unique index gates the INSERT race.
        cur.execute(
            """
            SELECT id, base_weight, support_propositions, support_axis_counts,
                   ontology_axis
            FROM Rel
            WHERE src_id = :s AND dst_id = :d AND predicate = :p
            FOR UPDATE
            """,
            s=src_id,
            d=dst_id,
            p=predicate,
        )
        row = cur.fetchone()

        if row is None:
            # No existing row. INSERT may race with another writer; the
            # unique constraint surfaces that as IntegrityError, the
            # retry decorator restarts from the SELECT.
            cur.execute(
                """
                INSERT INTO Rel (
                    src_id, dst_id, predicate, ontology_axis, base_weight,
                    support_propositions, support_axis_counts, source_id
                ) VALUES (
                    :s, :d, :p, :a, :bw,
                    :props, :counts, :sid
                )
                """,
                s=src_id,
                d=dst_id,
                p=predicate,
                a=ontology_axis,
                bw=base_weight,
                props=json.dumps([sp_hex]),
                counts=json.dumps({ontology_axis: 1}),
                sid=source_id,
            )
            return

        _, old_bw, old_props_lob, old_counts_lob, _old_axis = row
        support_props: list[str] = self._read_json(old_props_lob) or []
        axis_counts: dict[str, int] = self._read_json(old_counts_lob) or {}

        # Append proposition id without unbounded growth: de-dup and
        # cap at _MAX_SUPPORT_PROPS most-recent ids.
        if sp_hex not in support_props:
            support_props.append(sp_hex)
            if len(support_props) > _MAX_SUPPORT_PROPS:
                support_props = support_props[-_MAX_SUPPORT_PROPS:]

        axis_counts[ontology_axis] = int(axis_counts.get(ontology_axis, 0)) + 1

        # Most-frequent axis wins; ties broken by current-call assertion
        # (matches spec §5 step 5).
        best_axis = ontology_axis
        best_count = axis_counts[ontology_axis]
        for k, v in axis_counts.items():
            if int(v) > best_count:
                best_axis = k
                best_count = int(v)

        new_bw = (float(old_bw) + float(base_weight)) / 2.0

        cur.execute(
            """
            UPDATE Rel SET
                base_weight = :bw,
                ontology_axis = :a,
                support_propositions = :props,
                support_axis_counts = :counts,
                source_id = :sid,
                last_seen_at = SYSTIMESTAMP
            WHERE src_id = :s AND dst_id = :d AND predicate = :p
            """,
            bw=new_bw,
            a=best_axis,
            props=json.dumps(support_props),
            counts=json.dumps({k: int(v) for k, v in axis_counts.items()}),
            sid=source_id,
            s=src_id,
            d=dst_id,
            p=predicate,
        )

    def ingest_buffer(self, unit: IngestUnit) -> IngestWriteStats:
        """Write graph rows and ledger hashes for one extracted buffer atomically."""
        try:
            return self._ingest_buffer_once(unit)
        except oracledb.DatabaseError as e:
            raise GraphStoreError("failed to ingest buffer atomically") from e

    @retry(
        retry=retry_if_exception_type(oracledb.IntegrityError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.05, max=0.5),
        reraise=True,
    )
    def _ingest_buffer_once(self, unit: IngestUnit) -> IngestWriteStats:
        with self._conn() as c, c.cursor() as cur:
            try:
                entity_ids: dict[str, bytes] = {}
                for entity in unit.entities:
                    entity_ids[entity.name] = self._upsert_entity(
                        cur,
                        name=entity.name,
                        kind=entity.kind,
                        embedding=entity.embedding,
                        source_id=unit.source_id,
                    )

                rels = 0
                for prop in unit.propositions:
                    pid = self._upsert_proposition(
                        cur,
                        text=prop.text,
                        source_doc=unit.doc_id,
                        source_span=unit.section_path,
                        embedding=prop.embedding,
                        source_id=unit.source_id,
                    )
                    for triple in prop.triples:
                        base = min(1.0, 0.5 + 0.5 * float(triple.confidence))
                        self._upsert_rel(
                            cur,
                            entity_ids[triple.subject],
                            entity_ids[triple.object],
                            predicate=triple.predicate,
                            ontology_axis=triple.ontology_axis,
                            base_weight=base,
                            support_prop_id=pid,
                            source_id=unit.source_id,
                        )
                        rels += 1

                for span_hash in unit.span_hashes:
                    self._ledger_add(
                        cur,
                        span_hash,
                        doc_id=unit.doc_id,
                        section_path=unit.section_path,
                    )

                c.commit()
                return IngestWriteStats(
                    entities=len(unit.entities),
                    propositions=len(unit.propositions),
                    rels=rels,
                )
            except Exception:
                c.rollback()
                raise

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

    def vector_search_entities(
        self,
        *,
        query_vec: list[float],
        k: int,
        source_filter: str | None = None,
    ) -> list[dict]:
        with self._conn() as c, c.cursor() as cur:
            if source_filter is None:
                cur.execute(
                    """
                    SELECT id, name, VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                    FROM Entity ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                    """,
                    q=_vec(query_vec),
                    k=k,
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                    FROM Entity WHERE source_id = :sid
                    ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                    """,
                    q=_vec(query_vec),
                    sid=source_filter,
                    k=k,
                )
            return [
                {"id": r[0], "name": r[1], "distance": float(r[2])} for r in cur.fetchall()
            ]

    def vector_search_propositions(
        self,
        *,
        query_vec: list[float],
        k: int,
        source_filter: str | None = None,
    ) -> list[dict]:
        with self._conn() as c, c.cursor() as cur:
            if source_filter is None:
                cur.execute(
                    """
                    SELECT id, source_doc, source_span,
                           VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                    FROM Proposition ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                    """,
                    q=_vec(query_vec),
                    k=k,
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_doc, source_span,
                           VECTOR_DISTANCE(embedding, :q, COSINE) AS d
                    FROM Proposition WHERE source_id = :sid
                    ORDER BY d FETCH APPROX FIRST :k ROWS ONLY
                    """,
                    q=_vec(query_vec),
                    sid=source_filter,
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

    def list_sources(self) -> list[str]:
        """Return the distinct list of source_ids currently in Entity."""
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT DISTINCT source_id FROM Entity ORDER BY source_id")
            return [r[0] for r in cur.fetchall() if r[0] is not None]

    def pgql_subgraph(
        self,
        *,
        seed_ids: list[bytes],
        max_edges: int,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Return REL edges within one hop of any seed (i.e. seeds as either endpoint).

        Implementation notes:
        - VERTEX_ID() in 23ai returns a JSON composite key, not the raw RAW(16),
          which makes IN-list filtering against a bytes bind never match. We
          project `e.id` (the underlying property) instead — that comes back as
          plain bytes.
        - The :max_edges placeholder must be padded out as :m and bound separately
          because of the bind-parameter ordering rules around `FETCH FIRST`.
        - `source_filter`, when provided, is applied inside the MATCH WHERE on
          the edge's `source_id` property — GRAPH_TABLE doesn't allow a
          trailing WHERE on table columns, so the filter has to live inline.
        """
        if not seed_ids:
            return []
        # Bind names :s0..:sN appear twice in the SQL (e1.id IN ... OR e2.id IN ...).
        # oracledb resolves repeated bind-by-name correctly, so each name binds once.
        placeholders = ", ".join(f":s{i}" for i in range(len(seed_ids)))
        bind = {f"s{i}": sid for i, sid in enumerate(seed_ids)}
        bind["m"] = max_edges
        source_clause = ""
        if source_filter is not None:
            source_clause = " AND r.source_id = :sid"
            bind["sid"] = source_filter
        sql = f"""
            SELECT e1_id, e2_id, predicate, ontology_axis, base_weight,
                   support_propositions
            FROM GRAPH_TABLE (oragraph
              MATCH (e1) -[r IS REL]-> (e2)
              WHERE (e1.id IN ({placeholders})
                 OR e2.id IN ({placeholders})){source_clause}
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

    # ---------- IngestLedger ops ----------

    def ledger_has(self, span_hash: str) -> bool:
        """Returns True if the span hash is already in the IngestLedger."""
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM Ingest_Ledger WHERE span_hash = :h",
                h=span_hash,
            )
            return cur.fetchone() is not None

    def ledger_add(self, span_hash: str, *, doc_id: str, section_path: str) -> None:
        """Record a span hash as ingested. Idempotent on conflict."""
        with self._conn() as c, c.cursor() as cur:
            try:
                self._ledger_add(cur, span_hash, doc_id=doc_id, section_path=section_path)
                c.commit()
            except Exception:
                c.rollback()
                raise

    def clear_ingest_ledger(self) -> None:
        """Delete all ingest ledger rows through the graph-store boundary."""
        with self._conn() as c, c.cursor() as cur:
            try:
                cur.execute("DELETE FROM Ingest_Ledger")
                c.commit()
            except oracledb.DatabaseError as e:
                c.rollback()
                raise GraphStoreError("failed to clear ingest ledger") from e

    @staticmethod
    def _ledger_add(cur, span_hash: str, *, doc_id: str, section_path: str) -> None:
        cur.execute(
            """
            MERGE INTO Ingest_Ledger dst
            USING (
                SELECT :h AS span_hash, :d AS doc_id, :s AS section_path FROM dual
            ) src
            ON (dst.span_hash = src.span_hash)
            WHEN NOT MATCHED THEN
                INSERT (span_hash, doc_id, section_path)
                VALUES (src.span_hash, src.doc_id, src.section_path)
            """,
            h=span_hash,
            d=doc_id,
            s=section_path,
        )

    def iter_propositions_with_edges(
        self,
        *,
        source_filter: str | None = None,
    ) -> Iterator[tuple[dict, list[dict]]]:
        """Yield (prop_row, edge_rows) per Proposition.

        Each ``prop_row`` has id (bytes), text, source_doc, source_span,
        source_id. Each entry in ``edge_rows`` has predicate, ontology_axis,
        base_weight.

        The edge match looks up every REL whose ``support_propositions``
        JSON array contains this proposition's hex id. We use a single
        ``SELECT predicate, ontology_axis, base_weight, support_propositions
        FROM Rel`` and filter Python-side so behavior is uniform across the
        23ai builds we target (JSON_EXISTS path-expression support varies).
        Two-pass keeps the SQL simple at the cost of an O(props * rels)
        membership test; export is offline so the constant factor is fine.
        """
        where = ""
        bind: dict[str, object] = {}
        if source_filter is not None:
            where = " WHERE source_id = :sf"
            bind["sf"] = source_filter

        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT id, text, source_doc, source_span, source_id "
                f"FROM Proposition{where}",
                **bind,
            )
            prop_rows: list[dict] = []
            for row in cur:
                text_val = row[1].read() if hasattr(row[1], "read") else (row[1] or "")
                prop_rows.append(
                    {
                        "id": bytes(row[0]) if row[0] is not None else None,
                        "text": text_val,
                        "source_doc": row[2],
                        "source_span": row[3],
                        "source_id": row[4],
                    }
                )

            # Cache the full Rel table once and index by support_prop hex id.
            # For a graphified corpus this is bounded by total edges, which
            # is small relative to the dump cost itself.
            cur.execute(
                "SELECT predicate, ontology_axis, base_weight, support_propositions "
                "FROM Rel"
            )
            edges_by_prop: dict[str, list[dict]] = {}
            for e in cur:
                props = self._read_json(e[3]) or []
                edge = {
                    "predicate": e[0],
                    "ontology_axis": e[1],
                    "base_weight": float(e[2]),
                }
                for hex_id in props:
                    edges_by_prop.setdefault(str(hex_id), []).append(edge)

        for prop in prop_rows:
            pid = prop["id"]
            pid_hex = pid.hex() if isinstance(pid, bytes) else str(pid)
            yield prop, edges_by_prop.get(pid_hex, [])

    def fetch_propositions(
        self,
        ids: Iterable[bytes],
        *,
        source_filter: str | None = None,
    ) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ", ".join(f":i{i}" for i in range(len(ids)))
        bind: dict[str, object] = {f"i{i}": v for i, v in enumerate(ids)}
        where = f"id IN ({placeholders})"
        if source_filter is not None:
            where += " AND source_id = :sid"
            bind["sid"] = source_filter
        sql = (
            "SELECT id, text, source_doc, source_span FROM Proposition "
            f"WHERE {where}"
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
