"""Walk a folder, parse files by type, emit normalized Spans, buffer to ~token-bounded chunks.

Task 8 consumes the `Buffer` stream from `buffer_spans` and feeds each buffer's
text to the LLM extractor. Span boundaries respect document section structure
(markdown headings, PDF pages) so the extracted propositions stay localized.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from pypdf import PdfReader

# Rough estimator: characters per token. Used only for buffer sizing; the LLM
# never sees this number, so over/under-estimating by ~25% is fine. Real token
# counting via tiktoken would add a dep cost we don't need yet.
#
# Caveats (not a bug, just an FYI for tuners):
# - Source code tokenizes at ~3 chars/token, so code-heavy buffers under-estimate.
# - PDF extraction with mangled whitespace can over-pack characters.
# - Asian-language docs run ~1-2 chars/token.
# Buffers may therefore be up to ~2x the configured max_tokens for code/PDF
# content. Still safely under any modern LLM context window.
_TOKEN_APPROX = 4


@dataclass(slots=True)
class Span:
    doc_id: str
    section_path: str
    text: str
    span_id: str = ""

    @property
    def hash(self) -> str:
        """Stable content hash. Used by the IngestLedger to skip unchanged spans.

        Deliberately keyed on doc_id + section_path + text (not span_id), so a
        re-walk of the same folder produces the same hash for the same content
        even if span_id assignment changes.
        """
        h = hashlib.sha256()
        h.update(self.doc_id.encode())
        h.update(b"\x00")
        h.update(self.section_path.encode())
        h.update(b"\x00")
        h.update(self.text.encode())
        return h.hexdigest()


@dataclass(slots=True)
class Buffer:
    doc_id: str
    section_path: str
    text: str
    span_hashes: list[str] = field(default_factory=list)

    @property
    def approx_tokens(self) -> int:
        if not self.text:
            return 0
        return max(1, len(self.text) // _TOKEN_APPROX)


_SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".pdf"}


def walk_folder(root: Path | str) -> Iterator[Span]:
    """Recurse `root`, yielding Spans for every supported file type.

    Unsupported extensions are silently skipped. Files are sorted for
    deterministic ordering so the ingest ledger's span hashes are stable
    across runs.
    """
    root_path = Path(root)
    for p in sorted(root_path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        rel = p.relative_to(root_path).as_posix()
        suffix = p.suffix.lower()
        if suffix in {".md", ".markdown"}:
            yield from _parse_markdown(p, rel)
        elif suffix == ".txt":
            text = p.read_text(errors="replace").strip()
            if text:
                yield Span(doc_id=rel, section_path="", text=text)
        elif suffix == ".pdf":
            yield from _parse_pdf(p, rel)


def _parse_markdown(path: Path, rel: str) -> Iterator[Span]:
    md = MarkdownIt()
    tokens = md.parse(path.read_text(errors="replace"))
    section_stack: list[str] = []
    buf: list[str] = []

    def emit() -> Span | None:
        if not buf:
            return None
        text = "\n\n".join(buf).strip()
        buf.clear()
        if not text:
            return None
        return Span(doc_id=rel, section_path=" / ".join(section_stack), text=text)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            span = emit()
            if span:
                yield span
            inline = tokens[i + 1]
            level = int(tok.tag[1]) if tok.tag.startswith("h") else 1
            section_stack[level - 1 :] = [inline.content]
            i += 3
            continue
        if tok.type in ("inline", "fence", "code_block"):
            # `fence` is a triple-backtick block; `code_block` is the indented form.
            # Both carry their body in `tok.content` — capture so SQL/PL/SQL samples
            # in Oracle docs aren't silently dropped.
            buf.append(tok.content)
        i += 1

    span = emit()
    if span:
        yield span


def _parse_pdf(path: Path, rel: str) -> Iterator[Span]:
    reader = PdfReader(str(path))
    for n, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            yield Span(doc_id=rel, section_path=f"page-{n + 1}", text=text)


def buffer_spans(
    spans: Iterable[Span],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterator[Buffer]:
    """Group spans into token-bounded buffers, with overlap only at section boundaries.

    Buffer constraints:
    - All spans in a buffer come from the same (doc_id, section_path).
    - Buffer text size is approximately max_tokens (using char/4 heuristic).
    - When a buffer fills, the next buffer in the same section carries
      `overlap_tokens` worth of trailing text from the previous buffer.
    - When the section changes, the in-flight buffer is flushed with NO
      overlap into the new section.
    """
    cur_doc: str | None = None
    cur_section: str | None = None
    cur_text: list[str] = []
    cur_hashes: list[str] = []
    cur_tokens = 0

    def make_buffer() -> Buffer | None:
        if not cur_text:
            return None
        return Buffer(
            doc_id=cur_doc or "",
            section_path=cur_section or "",
            text="\n\n".join(cur_text).strip(),
            span_hashes=list(cur_hashes),
        )

    for span in spans:
        section_changed = (span.section_path != cur_section) or (span.doc_id != cur_doc)

        if section_changed:
            # Flush the in-flight buffer WITHOUT overlap into the new section.
            b = make_buffer()
            if b is not None:
                yield b
            cur_text = []
            cur_hashes = []
            cur_tokens = 0
            cur_doc = span.doc_id
            cur_section = span.section_path

        span_tokens = max(1, len(span.text) // _TOKEN_APPROX)

        # If adding this span would exceed the budget, flush WITH overlap.
        if cur_tokens + span_tokens > max_tokens and cur_text:
            b = make_buffer()
            if b is not None:
                yield b
            if overlap_tokens > 0:
                tail_chars = overlap_tokens * _TOKEN_APPROX
                joined = "\n\n".join(cur_text)
                tail = joined[-tail_chars:] if len(joined) > tail_chars else joined
                cur_text = [tail]
                cur_tokens = max(1, len(tail) // _TOKEN_APPROX)
                # Keep the trailing span's hash so the next buffer's span_hashes
                # acknowledges the overlap content. Without this, Task 8's ledger
                # logic would misjudge "all spans already ledgered" because the
                # carried-over span's hash would be dropped from the buffer.
                cur_hashes = cur_hashes[-1:]
            else:
                cur_text = []
                cur_tokens = 0
                cur_hashes = []

        cur_text.append(span.text)
        cur_hashes.append(span.hash)
        cur_tokens += span_tokens

    b = make_buffer()
    if b is not None:
        yield b
