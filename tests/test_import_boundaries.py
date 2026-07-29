"""Import-boundary gate for the two-pipeline split (plan Phase 6, gate 2).

``fawkes`` is the paper implementation; ``clinical_jepa`` is the modular
revision pipeline. They are separate packages because their schemas genuinely
differ, and neither may import the other. ``benchmarks`` exists to compare them
and is the only package allowed to import both.

Nothing here imports the packages under test. It parses them, so the gate stays
meaningful while the tree is half-built and while a package's dependencies may
not be installed. A package that does not exist yet is skipped, not failed —
but ``test_scan_covers_every_module_under_src`` makes sure the suite cannot go
green by quietly scanning nothing.

This replaces ``scripts/smoke_check.py::check_independence`` and
``test_suite.py::test_v5_v6_have_no_historical_package_imports``, which both
walk for ``graph_jepa_v2/v3/v4`` — packages that do not exist in this
repository, so they enforce nothing. Phase 7 deletes them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PAPER = "fawkes"
PIPELINE = "clinical_jepa"
BRIDGE = "benchmarks"

# The walk must not be able to report green by finding nothing. Two is the floor
# benchmarks/{__init__,llm_ranker}.py guarantees on its own, so this can never
# fail spuriously on a half-built tree — it exists to catch a walk that collapses
# to zero. The real coverage guarantee is the rglob equality below, which fails
# if the per-package walk misses even one module at any tree size.
MIN_MODULES = 2


def _import_roots(path: Path, package: str) -> set[str]:
    """Top-level package name of every import in one module.

    A relative import resolves to ``package``: ``from .x import y`` inside
    ``clinical_jepa`` can only ever reach ``clinical_jepa``, so it cannot cross
    a lineage boundary. Roots are matched exactly, which is why ``fawkes_core``
    (the old package, still in ``old_src``) does not read as ``fawkes``.
    """
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add(package)
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _scan(package: str) -> dict[Path, set[str]]:
    """Map every module under ``src/<package>`` to the roots it imports."""
    return {
        path: _import_roots(path, package)
        for path in sorted((SRC / package).rglob("*.py"))
    }


def _packages() -> list[str]:
    """Top-level packages present under ``src/``, however far built."""
    return sorted(
        entry.name
        for entry in SRC.iterdir()
        if entry.is_dir()
        and not entry.name.startswith((".", "__"))
        and any(entry.rglob("*.py"))
    )


def _require(package: str) -> dict[Path, set[str]]:
    if package not in _packages():
        pytest.skip(f"src/{package}/ not built yet")
    modules = _scan(package)
    assert modules, f"src/{package}/ exists but no module was scanned"
    return modules


def _offenders(modules: dict[Path, set[str]], forbidden: str) -> list[str]:
    return [
        str(path.relative_to(ROOT))
        for path, roots in modules.items()
        if forbidden in roots
    ]


def test_import_root_extraction_works(tmp_path):
    """The extractor itself, so a boundary test cannot pass by finding nothing.

    If _import_roots ever silently returns an empty set, every boundary
    assertion below becomes a tautology. This is the one test that would fail.
    """
    module = tmp_path / "sample.py"
    module.write_text(
        "import os\n"
        "import fawkes.model\n"
        "from clinical_jepa.graph import tensors\n"
        "from . import sibling\n"
        "from ..parent import thing\n"
        "import fawkes_core.schema as legacy\n",
        encoding="utf-8",
    )
    assert _import_roots(module, "benchmarks") == {
        "os",
        "fawkes",
        "clinical_jepa",
        "benchmarks",
        "fawkes_core",
    }


def test_clinical_jepa_does_not_import_fawkes():
    modules = _require(PIPELINE)
    assert not _offenders(modules, PAPER), (
        f"{PIPELINE} must not import {PAPER}: {_offenders(modules, PAPER)}"
    )


def test_fawkes_does_not_import_clinical_jepa():
    modules = _require(PAPER)
    assert not _offenders(modules, PIPELINE), (
        f"{PAPER} must not import {PIPELINE}: {_offenders(modules, PIPELINE)}"
    )


def test_benchmarks_is_the_only_cross_lineage_package():
    """Stated the other way round: whoever touches both lineages is benchmarks."""
    for package in _packages():
        roots = set().union(*_scan(package).values())
        both = {PAPER, PIPELINE} <= roots
        assert not both or package == BRIDGE, (
            f"src/{package}/ imports both {PAPER} and {PIPELINE}; "
            f"only {BRIDGE} may do that"
        )


def test_llm_ranker_imports_no_first_party_package():
    """The LLM client is shared by both benchmark drivers, so it stays neutral.

    benchmarks/ as a whole may import both lineages, but llm_ranker must import
    neither: it was extracted from the two duplicated evaluate_llm.py copies
    precisely so an LLM run needs no model package loaded.
    """
    roots = _import_roots(SRC / BRIDGE / "llm_ranker.py", BRIDGE)
    assert roots.isdisjoint(_packages()), f"llm_ranker imports first-party: {roots}"


def test_scan_covers_every_module_under_src():
    """The vacuity guard: per-package walks must see the whole tree."""
    packages = _packages()
    assert packages, "no packages found under src/"

    walked = {path for package in packages for path in _scan(package)}
    assert walked == set(SRC.rglob("*.py")), (
        f"per-package walk missed {set(SRC.rglob('*.py')) - walked}"
    )
    assert len(walked) >= MIN_MODULES, (
        f"scanned only {len(walked)} modules; expected at least {MIN_MODULES}"
    )
