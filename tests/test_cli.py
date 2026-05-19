from typer.testing import CliRunner

from oragraphrag.cli import app

runner = CliRunner()


def test_cli_help_lists_all_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for sub in ("graphify", "query", "bench", "init-db"):
        assert sub in out


def test_query_dry_run_does_not_touch_db_or_llm():
    """--dry-run must short-circuit so the CLI is testable without infra."""
    result = runner.invoke(app, ["query", "anything", "--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.stdout.lower()
    assert "anything" in result.stdout


def test_query_help_lists_required_args_and_options():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    # Question is a positional; --dry-run and --config are options.
    assert "question" in out
    assert "--dry-run" in out
    assert "--config" in out


def test_graphify_help_takes_folder_argument():
    result = runner.invoke(app, ["graphify", "--help"])
    assert result.exit_code == 0
    assert "folder" in result.stdout.lower()


def test_init_db_help_has_rebuild_flag():
    result = runner.invoke(app, ["init-db", "--help"])
    assert result.exit_code == 0
    assert "--rebuild" in result.stdout


def test_bench_help_shows_suite_and_systems_options():
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "--suite" in out
    assert "--systems" in out


def test_bench_command_fails_gracefully_on_missing_suite():
    """A missing --suite path must produce a clear error, not a traceback."""
    result = runner.invoke(
        app, ["bench", "--suite", "nope.jsonl", "--systems", "oragraphrag"]
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr if result.stderr else "")
    # The contract: the user must see WHY it didn't run.
    assert "not found" in combined.lower() or "nope.jsonl" in combined.lower()
    # Definitely not a Python-level traceback or KeyError leakage.
    assert "traceback" not in combined.lower()


def test_query_loads_config_from_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text("llm:\n  provider: ollama\n")
    # With --dry-run we don't actually call the LLM but the config load path runs.
    result = runner.invoke(
        app, ["query", "test", "--dry-run", "--config", str(cfg_file)]
    )
    assert result.exit_code == 0
    assert "dry run" in result.stdout.lower()
