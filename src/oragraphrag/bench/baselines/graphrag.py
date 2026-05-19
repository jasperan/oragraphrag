"""Microsoft GraphRAG baseline — stub until Task 16."""

from oragraphrag.config import Config


async def run(question: str, cfg: Config) -> dict:
    raise NotImplementedError(
        "GraphRAG baseline is wired in Task 16. Install the `graphrag` package "
        "and add a runner that points at the same Oracle 23ai vector store."
    )
