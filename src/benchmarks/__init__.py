"""Cross-lineage comparison harnesses.

This is the only package permitted to import both ``fawkes`` (the paper
implementation) and ``clinical_jepa`` (the modular revision pipeline). Neither
of those two may import the other; see ``tests/test_import_boundaries.py``,
which enforces that as a gate, and ``docs/LINEAGE.md`` for the old-name mapping.

Nothing is re-exported here on purpose. Importing ``benchmarks`` must not pull
in torch or either model package, so ``llm_ranker`` stays usable — and
importable — on its own.
"""
