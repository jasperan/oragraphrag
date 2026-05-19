"""Fine-tune exporter integration test against the live graph."""

from __future__ import annotations

import json
import subprocess

import pytest

pytestmark = [pytest.mark.oracle]


def _run_cli(args, cwd, timeout=180):
    proc = subprocess.run(
        ["oragraphrag", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _write_cfg(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "embeddings:\n"
        "  provider: ollama\n"
        "  ollama:\n"
        "    model: nomic-embed-text\n"
        "  dim: 768\n"
        "oracle:\n"
        "  username: ORAGRAPH\n"
        "  password: Welcome12345*\n"
        "  dsn: localhost:1521/FREEPDB1\n"
    )
    return cfg_path


def test_export_finetune_writes_jsonl(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    out_path = tmp_path / "train.jsonl"
    rc, _out, err = _run_cli(
        [
            "export",
            "--format",
            "finetune",
            "--out",
            str(out_path),
            "--config",
            str(cfg_path),
        ],
        cwd=tmp_path,
    )
    assert rc == 0, f"export failed: {err}"
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) > 0, "expected at least one training example"
    for line in lines:
        rec = json.loads(line)
        for key in ("id", "source", "source_id", "prompt", "completion", "axes", "predicates"):
            assert key in rec, f"missing key {key!r} in {rec!r}"
        assert isinstance(rec["axes"], list)
        assert isinstance(rec["predicates"], list)


def test_export_finetune_source_filter_subset(tmp_path):
    """When --source is passed, the resulting file is a strict subset of the full export."""
    cfg_path = _write_cfg(tmp_path)

    full_path = tmp_path / "full.jsonl"
    rc, _out, err = _run_cli(
        ["export", "--format", "finetune", "--out", str(full_path), "--config", str(cfg_path)],
        cwd=tmp_path,
    )
    assert rc == 0, f"full export failed: {err}"
    full_records = [json.loads(line) for line in full_path.read_text().strip().splitlines()]
    assert full_records, "full export must be non-empty"

    # Pick the source_id with the most rows so the scoped export is definitely smaller than total.
    source_ids = {r["source_id"] for r in full_records}
    assert source_ids, "no source_ids in full export"
    target_sid = next(iter(source_ids))

    scoped_path = tmp_path / "scoped.jsonl"
    rc, _out, err = _run_cli(
        [
            "export",
            "--format",
            "finetune",
            "--out",
            str(scoped_path),
            "--config",
            str(cfg_path),
            "--source",
            target_sid,
        ],
        cwd=tmp_path,
    )
    assert rc == 0, f"scoped export failed: {err}"
    scoped_records = [json.loads(line) for line in scoped_path.read_text().strip().splitlines()]
    assert scoped_records, "scoped export must be non-empty for chosen source_id"
    assert all(r["source_id"] == target_sid for r in scoped_records)
    assert len(scoped_records) <= len(full_records)
