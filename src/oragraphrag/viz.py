"""Visualization helpers for the notebook and the paper figures.

`render_subgraph_html` produces an interactive pyvis network used in the
demo notebook for showing per-query subgraph reweighting. `amplitude_heatmap_png`
produces a static matplotlib heatmap used in the notebook AND in the paper's
`paper/figures/edge_amplitude_heatmap.pdf` (matplotlib infers format from
the file extension).
"""

from __future__ import annotations

from pathlib import Path


def render_subgraph_html(
    edges: list[dict],
    *,
    activations: dict[bytes, float],
    out_path: Path,
) -> None:
    """Render a reweighted subgraph as an interactive pyvis HTML file.

    Edge thickness reflects `weight` (the post-amplitude value from
    `reweight_edges`); node size reflects `activation` (the PR score from
    `spreading_activation`). Nodes not present in `activations` get a
    small default size.
    """
    from pyvis.network import Network

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        directed=True,
        height="600px",
        width="100%",
        bgcolor="#111",
        font_color="#eee",
    )

    seen: set[bytes] = set()
    for e in edges:
        for n in (e["src"], e["dst"]):
            if n in seen:
                continue
            seen.add(n)
            act = activations.get(n, 0.0)
            net.add_node(
                n.hex(),
                label=n.hex()[:8],
                value=max(act, 0.01) * 100,
                color={"background": "#5af", "border": "#fff"},
            )
        net.add_edge(
            e["src"].hex(),
            e["dst"].hex(),
            value=max(e.get("weight", 0.0), 0.01) * 10,
            title=f"{e.get('predicate', '?')} ({e.get('ontology_axis', '?')})",
            label=e.get("ontology_axis", ""),
        )

    net.write_html(str(out_path), open_browser=False, notebook=False)


def amplitude_heatmap_png(
    amps_per_question: list[tuple[str, dict[str, float]]],
    *,
    out_path: Path,
) -> None:
    """Render a question-by-axis amplitude heatmap to a PNG or PDF file.

    Output format is inferred from `out_path`'s extension by matplotlib.
    Used in the demo notebook and as `paper/figures/edge_amplitude_heatmap.pdf`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = [q for q, _ in amps_per_question]
    axes_names = ["causal", "taxonomic", "temporal", "definitional", "exemplification"]
    mat = np.array(
        [[float(a.get(n, 0.0)) for n in axes_names] for _, a in amps_per_question]
    )

    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.4 * len(questions))))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(axes_names)))
    ax.set_xticklabels(axes_names, rotation=30, ha="right")
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=8)
    fig.colorbar(im, ax=ax, label="amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
