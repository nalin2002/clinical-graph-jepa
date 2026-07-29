"""The modular patch-based Graph-JEPA revision pipeline. This is **not** the
paper implementation — see the ``fawkes`` package.

Merged from ``fawkes_core`` (the v3/v4 base), ``graph_jepa_v5`` (no-note
variant), and ``graph_jepa_v6`` (localized-note variant). The two released
checkpoints are two configurations of one :class:`~clinical_jepa.model.GraphJEPA`
class, selected by ``ModelConfig.use_note_embeddings``.
"""

from .schema import EdgeType, NodeType, PatientGraph

from .config import Config

__all__ = ["Config", "EdgeType", "GraphJEPA", "NodeType", "PatientGraph"]


def __getattr__(name: str):
    if name == "GraphJEPA":
        from .model import GraphJEPA

        return GraphJEPA
    raise AttributeError(name)
