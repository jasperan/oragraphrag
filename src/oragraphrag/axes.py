"""Canonical descriptions used to seed the ontology axis vectors.

Each description is intentionally short and concrete so the embedding lands
in a stable region of the model's space. Changing these requires running
`oragraphrag init-db --rebuild` because the stored axis vectors will shift.

Ordering matters: ONTOLOGY_AXIS_NAMES is used directly by Task 9's reweighting
to iterate axes in a fixed sequence.
"""

ONTOLOGY_AXIS_NAMES: tuple[str, ...] = (
    "causal",
    "taxonomic",
    "temporal",
    "definitional",
    "exemplification",
)

AXIS_DESCRIPTIONS: dict[str, str] = {
    "causal": (
        "X causes Y. X leads to Y. X is the reason for Y. "
        "X triggers Y. If X then Y. X produces Y."
    ),
    "taxonomic": (
        "X is a kind of Y. X is a type of Y. X is a subclass of Y. "
        "X belongs to category Y. X inherits from Y."
    ),
    "temporal": (
        "X happens before Y. X precedes Y. X was introduced in version Y. "
        "X was deprecated in version Y. X replaces Y."
    ),
    "definitional": (
        "X is defined as Y. X means Y. The definition of X is Y. "
        "X refers to Y. X stands for Y."
    ),
    "exemplification": (
        "X is an example of Y. X illustrates Y. For instance, X demonstrates Y. "
        "An instance of X is Y. X shows Y in practice."
    ),
}
