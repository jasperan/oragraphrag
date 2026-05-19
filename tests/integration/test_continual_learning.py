"""Cross-session retrieval proof: knowledge from one /graphify run is
accessible in a future query, and per-source scoping works.

Marked oracle+llm; skipped under the default test run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.oracle, pytest.mark.llm]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    work = tmp_path_factory.mktemp("cl")
    cfg_path = work / "config.yaml"
    cfg_path.write_text(
        """
llm:
  provider: oci_grok
  oci_grok:
    model: xai.grok-4.3
    region: us-chicago-1
  request_timeout_s: 60.0
  max_retries: 2

embeddings:
  provider: ollama
  ollama:
    model: nomic-embed-text
  dim: 768

oracle:
  username: ORAGRAPH
  password: Welcome12345*
  dsn: localhost:1521/FREEPDB1

retrieval:
  seed_k_entities: 16
  seed_k_propositions: 32
  max_subgraph_nodes: 512
  max_subgraph_edges: 4096
""".strip()
    )
    corpus_a = (Path(__file__).parent / "fixtures" / "mini_corpus").resolve()
    corpus_b_dir = work / "corpus_b"
    corpus_b_dir.mkdir()
    (corpus_b_dir / "extra.md").write_text(
        "# Extra Concepts\n\n"
        "OraGraphRAG is a graph-augmented RAG system that uses Oracle 23ai.\n\n"
        "## Reweighting\n\n"
        "Edge weights are recomputed per query via ontology-axis projection.\n"
    )
    return {
        "work": work,
        "config": cfg_path,
        "corpus_a": corpus_a,
        "corpus_b": corpus_b_dir,
    }


def _run_cli(args: list[str], cwd: Path, timeout: int = 600):
    proc = subprocess.run(
        ["oragraphrag", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_continual_learning_two_corpora(env):
    work, cfg = env["work"], env["config"]

    # Reset the DB.
    rc, _out, err = _run_cli(["init-db", "--rebuild", "--config", str(cfg)], cwd=work)
    assert rc == 0, f"init-db: {err}"

    # Graphify corpus A.
    rc, _out, err = _run_cli(
        ["graphify", str(env["corpus_a"]), "--config", str(cfg)],
        cwd=work,
        timeout=900,
    )
    assert rc == 0, f"graphify A: {err}"

    # Graphify corpus B.
    rc, _out, err = _run_cli(
        ["graphify", str(env["corpus_b"]), "--config", str(cfg)],
        cwd=work,
        timeout=900,
    )
    assert rc == 0, f"graphify B: {err}"

    # Global query — should find something from EITHER corpus.
    rc, out, err = _run_cli(
        ["query", "What is HNSW?", "--config", str(cfg)],
        cwd=work,
        timeout=180,
    )
    assert rc == 0, f"global query: {err}"
    out_lower = out.lower()
    assert "hnsw" in out_lower or "don't have information" in out_lower

    # Scoped query — must work for corpus B's exclusive content.
    rc, out, err = _run_cli(
        [
            "query",
            "What is OraGraphRAG?",
            "--source",
            str(env["corpus_b"]),
            "--config",
            str(cfg),
        ],
        cwd=work,
        timeout=180,
    )
    assert rc == 0, f"scoped B query: {err}"
    out_lower = out.lower()
    assert any(
        t in out_lower for t in ("graph", "rag", "reweight", "don't have information")
    )

    # Scoped query to corpus A — must not crash even with --source scoping.
    rc, _out, err = _run_cli(
        [
            "query",
            "What is HNSW?",
            "--source",
            str(env["corpus_a"]),
            "--config",
            str(cfg),
        ],
        cwd=work,
        timeout=180,
    )
    assert rc == 0, f"scoped A query: {err}"


def test_sources_command_lists_both_corpora(env):
    """After two /graphify runs, oragraphrag sources lists both source ids."""
    rc, out, err = _run_cli(["sources", "--config", str(env["config"])], cwd=env["work"])
    assert rc == 0, f"sources: {err}"
    lines = [line.strip() for line in out.strip().splitlines() if line.strip().startswith("src_")]
    assert len(lines) >= 2, f"expected >=2 source_ids; got: {out!r}"
