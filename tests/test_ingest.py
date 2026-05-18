from pathlib import Path

from oragraphrag.ingest import Span, buffer_spans, walk_folder


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_walk_folder_emits_spans_for_md(tmp_path):
    _write(tmp_path / "a.md", "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n")
    spans = list(walk_folder(tmp_path))
    docs = {s.doc_id for s in spans}
    assert "a.md" in docs
    sections = {s.section_path for s in spans}
    assert any("Title" in p for p in sections)


def test_walk_folder_skips_unsupported_extensions(tmp_path):
    _write(tmp_path / "a.md", "# Title\n\nbody.\n")
    _write(tmp_path / "skip.bin", "binary")
    _write(tmp_path / "skip.png", "fake png")
    spans = list(walk_folder(tmp_path))
    docs = {s.doc_id for s in spans}
    assert "a.md" in docs
    assert "skip.bin" not in docs
    assert "skip.png" not in docs


def test_walk_folder_handles_plain_txt(tmp_path):
    _write(tmp_path / "notes.txt", "Just a flat text file with no sections.\n")
    spans = list(walk_folder(tmp_path))
    assert any(s.doc_id == "notes.txt" for s in spans)


def test_walk_folder_recurses_into_subdirs(tmp_path):
    _write(tmp_path / "sub" / "deep.md", "# Deep\n\ncontent\n")
    spans = list(walk_folder(tmp_path))
    docs = {s.doc_id for s in spans}
    assert "sub/deep.md" in docs


def test_walk_folder_empty_dir_returns_no_spans(tmp_path):
    spans = list(walk_folder(tmp_path))
    assert spans == []


def test_buffer_spans_respects_token_budget():
    spans = [
        Span(doc_id="d", section_path="s", text="word " * 200, span_id=str(i))
        for i in range(20)
    ]
    bufs = list(buffer_spans(spans, max_tokens=500, overlap_tokens=50))
    # Each buffer should be at-or-near the budget, never wildly over.
    assert all(b.approx_tokens <= 700 for b in bufs)
    # 20 spans * 200 tokens = 4000 tokens; should produce 8+ buffers at ~500 budget.
    assert len(bufs) >= 4


def test_buffer_spans_does_not_overlap_across_sections():
    spans = [
        Span(doc_id="d", section_path="s1", text="alpha " * 100, span_id="1"),
        Span(doc_id="d", section_path="s2", text="beta " * 100, span_id="2"),
    ]
    bufs = list(buffer_spans(spans, max_tokens=1000, overlap_tokens=50))
    sections = {b.section_path for b in bufs}
    assert sections == {"s1", "s2"}
    # No buffer should contain text from both sections.
    for b in bufs:
        if b.section_path == "s1":
            assert "beta" not in b.text
        if b.section_path == "s2":
            assert "alpha" not in b.text


def test_buffer_spans_empty_input():
    assert list(buffer_spans([], max_tokens=500, overlap_tokens=50)) == []


def test_span_hash_is_stable_and_keys_on_content():
    s1 = Span(doc_id="d", section_path="s", text="hello", span_id="1")
    s2 = Span(doc_id="d", section_path="s", text="hello", span_id="2")
    s3 = Span(doc_id="d", section_path="s", text="goodbye", span_id="1")
    # span_id is not part of the hash; only doc_id + section_path + text are.
    assert s1.hash == s2.hash
    assert s1.hash != s3.hash


def test_buffer_carries_span_hashes():
    spans = [
        Span(doc_id="d", section_path="s", text="a", span_id="1"),
        Span(doc_id="d", section_path="s", text="b", span_id="2"),
    ]
    bufs = list(buffer_spans(spans, max_tokens=1000, overlap_tokens=0))
    assert len(bufs) == 1
    assert len(bufs[0].span_hashes) == 2
