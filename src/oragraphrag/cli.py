"""Typer CLI: init-db | graphify | query | bench.

The CLI is intentionally thin: it parses args, loads config, and delegates
to the pipeline modules. Heavy lifting lives in pipeline_ingest.py and
pipeline_query.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from oragraphrag.config import Config

app = typer.Typer(
    help="OraGraphRAG — Oracle-backed graph-augmented RAG.",
    no_args_is_help=True,
)
console = Console()


def _load_config(path: Path | None) -> Config:
    """Load Config from explicit path, then ./config.yaml, then defaults."""
    if path is not None:
        if not path.exists():
            console.print(f"[red]config file not found: {path}[/red]")
            raise typer.Exit(1)
        return Config.from_yaml(path)
    default = Path("config.yaml")
    if default.exists():
        return Config.from_yaml(default)
    return Config()


@app.command("init-db")
def init_db_cmd(
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml."),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop and recreate schema."),  # noqa: B008
) -> None:
    """Create tables, HNSW indexes, the property graph, and ontology axis rows."""
    from oragraphrag.embed import Embedder, build_axis_vectors
    from oragraphrag.graph import GraphStore

    cfg = _load_config(config)

    from oragraphrag.embed_backends import build_embed_backend

    async def _go() -> None:
        store = GraphStore(cfg)
        store.connect()
        try:
            # Use the real configured embedder so axis vectors are meaningful.
            # Task 9's reweighting depends on these being real embeddings of
            # the canonical axis descriptions, not zero vectors.
            emb = Embedder(cfg, backend=build_embed_backend(cfg, store))
            axes = await build_axis_vectors(emb)
            store.init_db(
                rebuild=rebuild,
                axis_vectors={k: v.tolist() for k, v in axes.items()},
            )
        finally:
            store.close()

    asyncio.run(_go())
    console.print("[green]init-db complete[/green]")


@app.command("graphify")
def graphify_cmd(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder to ingest."),  # noqa: B008
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml."),  # noqa: B008
    reextract: bool = typer.Option(  # noqa: B008
        False, "--reextract", help="Clear the ledger before ingesting."
    ),
) -> None:
    """Walk a folder and ingest into the property graph."""
    from oragraphrag.embed import Embedder
    from oragraphrag.extract import Extractor
    from oragraphrag.graph import GraphStore
    from oragraphrag.ingest import buffer_spans, walk_folder
    from oragraphrag.llm import LLM
    from oragraphrag.pipeline_ingest import IngestPipeline

    cfg = _load_config(config)
    store = GraphStore(cfg)
    store.connect()
    try:
        if reextract:
            with store._conn() as c, c.cursor() as cur:
                cur.execute("DELETE FROM Ingest_Ledger")
                c.commit()

        # Task 13 ships the real embedding backend. Until then, the CLI
        # imports a per-cfg backend selector. For test isolation we
        # tolerate the absence of the backends module.
        try:
            from oragraphrag.embed_backends import build_embed_backend
        except ImportError:
            console.print(
                "[yellow]Note: embed_backends module not present yet (Task 13). "
                "graphify needs a real embedder backend.[/yellow]"
            )
            raise typer.Exit(1) from None

        emb = Embedder(cfg, backend=build_embed_backend(cfg, store))
        spans = walk_folder(folder)
        bufs = list(
            buffer_spans(
                spans,
                max_tokens=cfg.ingest.span_max_tokens,
                overlap_tokens=cfg.ingest.section_overlap_tokens,
            )
        )

        async def _run_ingest() -> dict:
            async with LLM(cfg) as llm:
                extractor = Extractor(llm)
                pipeline = IngestPipeline(
                    cfg=cfg, graph=store, embedder=emb, extractor=extractor
                )
                return await pipeline.run(bufs)

        stats = asyncio.run(_run_ingest())
        console.print_json(json.dumps(stats))
    finally:
        store.close()


@app.command("query")
def query_cmd(
    question: str = typer.Argument(..., help="The question to answer."),  # noqa: B008
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml."),  # noqa: B008
    dry_run: bool = typer.Option(  # noqa: B008
        False,
        "--dry-run",
        help="Print what would be queried without touching the DB or LLM.",
    ),
) -> None:
    """Answer a question from the indexed corpus."""
    cfg = _load_config(config)
    if dry_run:
        console.print(f"dry run: would query [bold]{question!r}[/bold]")
        console.print(f"config: provider={cfg.llm.provider}, dim={cfg.embeddings.dim}")
        raise typer.Exit(0)

    from oragraphrag.embed import Embedder, build_axis_vectors
    from oragraphrag.graph import GraphStore
    from oragraphrag.llm import LLM
    from oragraphrag.pipeline_query import QueryPipeline

    try:
        from oragraphrag.embed_backends import build_embed_backend
    except ImportError:
        console.print(
            "[yellow]Note: embed_backends module not present yet (Task 13). "
            "query needs a real embedder backend.[/yellow]"
        )
        raise typer.Exit(1) from None

    store = GraphStore(cfg)
    store.connect()
    try:
        emb = Embedder(cfg, backend=build_embed_backend(cfg, store))

        async def _run_query() -> None:
            axes = await build_axis_vectors(emb)
            async with LLM(cfg) as llm:
                pipeline = QueryPipeline(
                    cfg=cfg,
                    graph=store,
                    embedder=emb,
                    llm=llm,
                    axis_vectors=axes,
                )
                result = await pipeline.query(question)
                console.print(result.answer.text)
                if result.answer.citations:
                    console.print("\n[dim]Citations:[/dim]")
                    for c in result.answer.citations:
                        console.print(f"  - {c.source_doc}#{c.source_span}")

        asyncio.run(_run_query())
    finally:
        store.close()


@app.command("bench")
def bench_cmd(
    suite: str = typer.Option(..., "--suite", help="Path to the bench suite JSONL."),  # noqa: B008
    systems: str = typer.Option(  # noqa: B008
        "oragraphrag", "--systems", help="Comma-separated baseline names."
    ),
    limit: int | None = typer.Option(  # noqa: B008
        None, "--limit", help="Cap the number of questions for smoke runs."
    ),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml."),  # noqa: B008
) -> None:
    """Run the benchmark harness across one or more baselines."""
    try:
        from oragraphrag.bench.runner import run_suite
    except ImportError:
        console.print(
            "[yellow]bench harness not implemented yet (Task 15). "
            "Re-run after Task 15 lands.[/yellow]"
        )
        raise typer.Exit(1) from None

    cfg = _load_config(config)
    systems_list = [s.strip() for s in systems.split(",") if s.strip()]
    if not Path(suite).exists():
        console.print(f"[red]bench suite not found: {suite}[/red]")
        raise typer.Exit(1)
    result = asyncio.run(
        run_suite(cfg, suite=suite, systems=systems_list, limit=limit)
    )
    console.print_json(json.dumps(result))
