from oragraphrag.config import Config


def test_config_loads_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    assert cfg.llm.provider == "oci_grok"
    assert cfg.embeddings.dim == 384
    assert cfg.retrieval.amplitude.alpha == 8.0
    assert cfg.retrieval.amplitude.beta == 0.0


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("OGR__LLM__PROVIDER", "ollama")
    monkeypatch.setenv("OGR__RETRIEVAL__AMPLITUDE__ALPHA", "4.5")
    cfg = Config()
    assert cfg.llm.provider == "ollama"
    assert cfg.retrieval.amplitude.alpha == 4.5


def test_config_yaml_load(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "llm:\n  provider: ollama\nembeddings:\n  dim: 768\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = Config.from_yaml(cfg_file)
    assert cfg.llm.provider == "ollama"
    assert cfg.embeddings.dim == 768


def test_config_yaml_null_section_falls_back_to_defaults(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm: null\nembeddings:\n  dim: 768\n")
    monkeypatch.chdir(tmp_path)
    cfg = Config.from_yaml(cfg_file)
    assert cfg.llm.provider == "oci_grok"  # back to default
    assert cfg.embeddings.dim == 768       # explicit override applied


def test_config_lowercase_env_var_works(monkeypatch):
    monkeypatch.setenv("ogr__llm__provider", "ollama")
    cfg = Config()
    assert cfg.llm.provider == "ollama"


def test_env_outranks_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "oracle:\n  dsn: attacker.example:1521/X\n  password: planted\n"
        "llm:\n  provider: ollama\n"
    )
    monkeypatch.setenv("OGR__ORACLE__DSN", "operator.example:1521/Y")
    monkeypatch.setenv("OGR__ORACLE__PASSWORD", "operator-secret")
    cfg = Config.from_yaml(cfg_file)
    assert cfg.oracle.dsn == "operator.example:1521/Y"
    assert cfg.oracle.password == "operator-secret"
    assert cfg.llm.provider == "ollama"  # yaml still applies where env is silent
