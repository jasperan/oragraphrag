from oragraphrag.viz import amplitude_heatmap_png, render_subgraph_html


def test_render_subgraph_writes_html(tmp_path):
    edges = [
        {
            "src": b"\x01",
            "dst": b"\x02",
            "weight": 0.7,
            "predicate": "rel",
            "ontology_axis": "causal",
            "support_propositions": [],
        }
    ]
    activations = {b"\x01": 0.5, b"\x02": 0.3}
    out = tmp_path / "g.html"
    render_subgraph_html(edges, activations=activations, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 500  # non-trivial HTML
    content = out.read_text()
    assert "html" in content.lower()


def test_render_subgraph_handles_empty_edges(tmp_path):
    out = tmp_path / "g.html"
    render_subgraph_html([], activations={}, out_path=out)
    assert out.exists()


def test_render_subgraph_with_missing_activations_uses_default(tmp_path):
    """A node that's not in activations should still render with a default value."""
    edges = [
        {
            "src": b"\x01",
            "dst": b"\x02",
            "weight": 0.5,
            "predicate": "rel",
            "ontology_axis": "causal",
            "support_propositions": [],
        }
    ]
    # No activations for either node.
    out = tmp_path / "g_default_act.html"
    render_subgraph_html(edges, activations={}, out_path=out)
    assert out.exists()


def test_amplitude_heatmap_writes_png(tmp_path):
    amps = [
        (
            "q1",
            {
                "causal": 0.9,
                "taxonomic": 0.1,
                "temporal": 0.2,
                "definitional": 0.3,
                "exemplification": 0.1,
            },
        ),
        (
            "q2",
            {
                "causal": 0.3,
                "taxonomic": 0.8,
                "temporal": 0.5,
                "definitional": 0.4,
                "exemplification": 0.2,
            },
        ),
    ]
    out = tmp_path / "h.png"
    amplitude_heatmap_png(amps, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000  # actual image content


def test_amplitude_heatmap_handles_single_question(tmp_path):
    amps = [
        (
            "q1",
            {
                "causal": 0.9,
                "taxonomic": 0.1,
                "temporal": 0.2,
                "definitional": 0.3,
                "exemplification": 0.1,
            },
        ),
    ]
    out = tmp_path / "h_single.png"
    amplitude_heatmap_png(amps, out_path=out)
    assert out.exists()


def test_amplitude_heatmap_creates_parent_dirs(tmp_path):
    """If the output path's parent directory doesn't exist, it should be created."""
    out = tmp_path / "subdir" / "deeper" / "h.png"
    amps = [
        (
            "q",
            {
                "causal": 0.5,
                "taxonomic": 0.5,
                "temporal": 0.5,
                "definitional": 0.5,
                "exemplification": 0.5,
            },
        )
    ]
    amplitude_heatmap_png(amps, out_path=out)
    assert out.exists()


def test_amplitude_heatmap_can_emit_pdf(tmp_path):
    """The paper figure target is a PDF; matplotlib infers format from extension."""
    out = tmp_path / "h.pdf"
    amps = [
        (
            "q",
            {
                "causal": 0.9,
                "taxonomic": 0.1,
                "temporal": 0.2,
                "definitional": 0.3,
                "exemplification": 0.1,
            },
        )
    ]
    amplitude_heatmap_png(amps, out_path=out)
    assert out.exists()
    # PDF magic bytes are %PDF.
    head = out.read_bytes()[:5]
    assert head.startswith(b"%PDF")
