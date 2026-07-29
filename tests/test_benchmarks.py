"""Phase 6 gates — `benchmarks`, the cross-lineage comparison package.

Two modules are gated here: `vs_llm.py` (the merge of the two
`evaluate_llm.py` copies) and `vs_fawkes.py` (the merge of the two
`evaluate_loo_v12_jepa_llm.py` copies, with the v12 LOO baseline arm replaced by
the v16 paper checkpoint — see that module's docstring for why).

**The LLM arm itself is not gated, and cannot be.** It calls a remote
chat-completions endpoint; the response is non-deterministic, there is no pinned
baseline for it, and running it needs a paid API key. Plan Phase 6 gate 1 and
§10's risk table both say so explicitly. What *is* gated is everything
deterministic around it: which edges become queries, the exact prompt text, the
parsing of a ranking out of a reply, the aggregation of ranks into metrics, and
both model arms' numbers. The only untested code in the LLM path is
`ChatRanker.rank`'s HTTP call.

The strongest gate in the file is `test_fawkes_arm_reproduces_paper_test_split`,
which is exact rather than toleranced: Phase 0 measured a delta of `0.000e+00`
against `baseline/paper_loo_testsplit.json`, so anything else is a regression.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import astuple

import pytest
import torch

from conftest import BASELINE, DATA, ROOT, requires_checkpoints, requires_data

from benchmarks import vs_fawkes, vs_llm
from clinical_jepa.encoders import MockEncoder
from clinical_jepa.evaluate import leave_one_out_recovery
from clinical_jepa.graph.builders import adapt_mimic_subkg
from clinical_jepa.graph.tensors import PatientGraphDataset
from clinical_jepa.schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX
from clinical_jepa.train import loop
from fawkes.evaluate import _load_graphs

from fawkes_core.data import adapt_mimic_subkg as old_adapt_mimic_subkg
from graph_jepa_v6 import evaluate as old_evaluate
from graph_jepa_v6 import evaluate_llm as old_llm
from graph_jepa_v6 import evaluate_loo_v12_jepa_llm as old_three_way
from graph_jepa_v6 import training as old_training
from graph_jepa_v6.data import PatientGraphDataset as OldPatientGraphDataset

DEVICE = torch.device("cpu")
PAPER_CKPT = ROOT / "models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt"
V6_CKPT = ROOT / "models/v6_with_note/graph_jepa_v6.pt"
BENCHMARKS_SRC = ROOT / "src/benchmarks"

# Enough test-split admissions for the differential to cover every relation the
# clinical_jepa arm ranks, without paying for all 400 twice.
DIFFERENTIAL_GRAPHS = 120

requires_paper_checkpoint = pytest.mark.skipif(
    not PAPER_CKPT.exists(),
    reason="released paper checkpoint not present in the working tree",
)


@pytest.fixture(scope="module")
def fawkes_run():
    """Run the v16 baseline arm once; two gates read from it.

    Module-scoped because it parses the whole 234 MB JSONL and converts all
    4,000 records before it can take the seeded split.
    """
    raw, demographics = _load_graphs(DATA, None)
    metrics, records, cfg = vs_fawkes.fawkes_arm(
        str(PAPER_CKPT),
        raw,
        demographics,
        DEVICE,
        cap=40000,
    )
    return raw, metrics, records, cfg


# --------------------------------------------------------------------------- #
# Gate 1: the fawkes arm reproduces the published test-split numbers, exactly.
# --------------------------------------------------------------------------- #
@requires_paper_checkpoint
@requires_data
def test_fawkes_arm_reproduces_paper_test_split(fawkes_run):
    """Exact, not `1e-6`. Phase 0 recorded a delta of `0.000e+00`.

    This is the numeric gate the v12 version of this comparison could never
    have: its checkpoint is absent, so its arm could not run at all. Every
    per-relation row is compared too, because the headline MRR alone would not
    notice two relations trading places.
    """
    _raw, metrics, records, cfg = fawkes_run
    expected = json.loads((BASELINE / "paper_loo_testsplit.json").read_text())

    assert (cfg.seed, cfg.test_frac) == (42, 0.1)
    assert len(records) == 400
    assert metrics["n"] == expected["n"] == 8283
    for key in ("mrr", "hits1", "hits3", "hits10"):
        assert metrics[key] == expected[key], f"{key}: {metrics[key]!r}"

    assert [row["rel"] for row in metrics["per_rel"]] == [
        row["rel"] for row in expected["per_rel"]
    ]
    for got, want in zip(metrics["per_rel"], expected["per_rel"]):
        assert got == want, got["rel"]


# --------------------------------------------------------------------------- #
# Gate 2: the clinical_jepa arm, differentially against old_src.
# --------------------------------------------------------------------------- #
@requires_paper_checkpoint
@requires_checkpoints
@requires_data
def test_clinical_jepa_arm_matches_old_pipeline(fawkes_run):
    """New pipeline vs `graph_jepa_v6`'s, in one process, on the same records.

    `baseline/v{5,6}_loo.json` are synthetic-graph runs, so there is no pinned
    oracle for this arm on this dataset; the differential is the gate. Each side
    builds its own `PatientGraph`s with its own adapter and its own dataset
    class, so a drift anywhere between the raw JSON and the metric dict shows up
    here.

    Both sides share one `MockEncoder`: node features are held constant so the
    comparison is of the pipeline, not of an encoder download. The metric values
    are therefore not meaningful — the equality is.
    """
    raw, _metrics, records, _cfg = fawkes_run
    records = records[:DIFFERENTIAL_GRAPHS]

    model, cfg = loop.load_model_checkpoint(str(V6_CKPT), DEVICE)
    model.eval()
    encoder = MockEncoder(dim=cfg.model.base_in_dim)
    _graphs, data_list = vs_fawkes.clinical_jepa_pipeline(
        raw,
        records,
        encoder,
        cfg,
        source=str(DATA),
    )
    new = leave_one_out_recovery(
        model,
        data_list,
        cfg,
        DEVICE,
        cap=40000,
        candidate_mode="same-type",
    )

    old_model, old_cfg = old_training.load_model_checkpoint(str(V6_CKPT), DEVICE)
    old_model.eval()
    old_dataset = OldPatientGraphDataset(
        [
            old_adapt_mimic_subkg(raw[index], source_path=f"{DATA}:{index + 1}")
            for index in records
        ],
        encoder,
        use_note_embeddings=old_cfg.model.use_note_embeddings,
        note_embedding_dim=old_cfg.model.note_embedding_dim,
        note_ground_by=old_cfg.model.note_ground_by,
    )
    old = old_evaluate.leave_one_out_recovery(
        old_model,
        [old_dataset[idx] for idx in range(len(old_dataset))],
        old_cfg,
        DEVICE,
        cap=40000,
        candidate_mode="same-type",
    )

    assert new == old
    # Not vacuous: the slice has to actually rank something, across relations.
    assert new["n"] > 0
    assert len(new["per_rel"]) > 1


# --------------------------------------------------------------------------- #
# The driver itself, with the one non-deterministic dependency stubbed out.
# --------------------------------------------------------------------------- #
@requires_paper_checkpoint
@requires_checkpoints
@requires_data
def test_run_drives_all_three_arms(tmp_path, monkeypatch):
    """The only end-to-end exercise of `run`, and the only cover for its LLM branch.

    A real run of that branch costs money, so a bug in it would surface at the
    worst possible moment. The LLM's *answers* stay ungated — the stub returns a
    fixed ranking — but the loop around them does not: the paired
    `clinical_jepa` rank, the incremental JSONL, the summary block and the
    payload are all real.

    `--cap 1` reduces both model arms to one query each; gate 1 is what checks
    their numbers. The encoder is mocked because the released `clinical_jepa`
    config asks for SapBERT and a cold cache would download it.
    """

    class StubRanker:
        def __init__(self, **_kwargs):
            pass

        def rank(self, prompt):
            assert prompt.startswith("You are ranking candidate target nodes")
            return '{"ranking":[2,1]}', {
                "prompt_tokens": 10, "completion_tokens": 3,
                "total_tokens": 13, "finish_reason": "stop",
            }

    monkeypatch.setattr(
        vs_fawkes, "build_checkpoint_encoder",
        lambda cfg, _cache: MockEncoder(dim=cfg.model.base_in_dim),
    )
    monkeypatch.setattr(vs_fawkes, "ChatRanker", StubRanker)
    monkeypatch.setattr(vs_fawkes, "_api_key", lambda provider: "stub")

    output = tmp_path / "vs_fawkes.json"
    records_output = tmp_path / "records.jsonl"
    payload = vs_fawkes.run(
        vs_fawkes.build_arg_parser().parse_args([
            "--cap", "1", "--max-queries", "4", "--llm-model", "stub/model",
            "--output", str(output), "--records-output", str(records_output),
        ])
    )

    assert payload["population"] == {
        "records": 4000, "test_split_graphs": 400, "split_seed": 42,
        "test_frac": 0.1, "fawkes_edges": 1, "clinical_jepa_edges": 1,
        "llm_queries": 4,
    }
    assert len(payload["records"]) == 4
    assert records_output.read_text().count("\n") == 4
    assert json.loads(output.read_text())["population"] == payload["population"]
    # Both readings of the sampled queries are present; without the paired one
    # the LLM row would have nothing comparable beside it.
    assert payload["clinical_jepa_sampled"]["n"] == payload["llm"]["n"] == 4
    assert payload["llm_parse_failures"] == 0


# --------------------------------------------------------------------------- #
# The vs_llm port: sampling, prompts and ranks, differentially.
# --------------------------------------------------------------------------- #
SYNTHETIC_ARGV = ["--data", "synthetic", "--synthetic-graphs", "32"]
SAMPLING = dict(
    samples_per_relation=5,
    candidate_mode="same-type",
    max_candidates=8,
    seed=0,
)


def _synthetic_side(training, dataset_cls, load_checkpoint):
    """Build one lineage's (graphs, data_list, model, cfg) from seeded synthetic graphs."""
    parser = argparse.ArgumentParser()
    training.add_data_args(parser)
    args = parser.parse_args(SYNTHETIC_ARGV)
    model, cfg = load_checkpoint(str(V6_CKPT), DEVICE)
    model.eval()
    cfg.train.synthetic_graphs = args.synthetic_graphs
    graphs = training.build_graphs(args, cfg)
    dataset = dataset_cls(
        graphs,
        MockEncoder(dim=cfg.model.base_in_dim),
        use_note_embeddings=cfg.model.use_note_embeddings,
        note_embedding_dim=cfg.model.note_embedding_dim,
        note_ground_by=cfg.model.note_ground_by,
    )
    return graphs, [dataset[idx] for idx in range(len(dataset))], model, cfg


@requires_checkpoints
def test_query_sampling_prompts_and_ranks_match_old_evaluate_llm():
    """Everything the LLM comparison does before and after the API call.

    `collect_recovery_queries` picks the edges, `build_prompt` writes the text
    the model sees, `jepa_rank_query` produces the row the LLM is compared
    against. None of the three touches the network, so all three are exact
    against `graph_jepa_v6/evaluate_llm.py`.
    """
    graphs, data_list, model, cfg = _synthetic_side(
        loop, PatientGraphDataset, loop.load_model_checkpoint
    )
    old_graphs, old_data_list, old_model, old_cfg = _synthetic_side(
        old_training, OldPatientGraphDataset, old_training.load_model_checkpoint
    )

    queries = vs_llm.collect_recovery_queries(data_list, cfg, **SAMPLING)
    old_queries = old_llm.collect_recovery_queries(old_data_list, old_cfg, **SAMPLING)

    assert queries, "no queries sampled; the differential would be vacuous"
    assert [astuple(q) for q in queries] == [astuple(q) for q in old_queries]

    prompt_kwargs = dict(context_mode="full", max_context_edges=120)
    for query, old_query in zip(queries, old_queries):
        assert vs_llm.build_prompt(
            graphs[query.graph_index], data_list[query.graph_index],
            cfg, query, **prompt_kwargs,
        ) == old_llm.build_prompt(
            old_graphs[old_query.graph_index], old_data_list[old_query.graph_index],
            old_cfg, old_query, **prompt_kwargs,
        )
        assert vs_llm.jepa_rank_query(
            model, data_list[query.graph_index], query, cfg, DEVICE
        ) == old_llm.jepa_rank_query(
            old_model, old_data_list[old_query.graph_index], old_query, old_cfg, DEVICE
        )


@pytest.mark.parametrize(
    "reply",
    [
        '{"ranking":[3,1,2]}',
        'sure, here you go:\n{"ranked_candidates": ["2", "1"]}\nhope that helps',
        "I think 2 then 1 then 3",
        "no numbers at all",
        '{"ranking":[99,2]}',
        "",
    ],
)
def test_parse_ranking_matches_old(reply):
    """The reply parser is the one part of the LLM path that must not drift:
    a silent change here rewrites the LLM's score without touching the model."""
    assert vs_llm.parse_ranking(reply, 3) == old_llm.parse_ranking(reply, 3)


def test_summarize_merges_two_identical_copies():
    """`_summarize` existed twice in `old_src`, annotated with two different
    dataclasses. One copy now serves both benchmark drivers; this asserts it
    agrees with each of the two it replaced."""
    ranks = [(1, 4), (2, 1), (3, 3), (1, 2), (7, 7)]
    relations = ["MANAGED_FOR", "MANAGED_FOR", "INDICATES", "INDICATES", "CONFIRMS"]
    shared = dict(
        graph_index=0, edge_index=0, source="a", target="b",
        candidates=["a", "b", "c", "d"],
        llm_parse_ok=True, llm_parse_complete=True,
        prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=0,
        finish_reason="stop", llm_response="",
    )
    new = [
        vs_llm.QueryResult(relation=rel, jepa_rank=j, llm_rank=m, **shared)
        for rel, (j, m) in zip(relations, ranks)
    ]
    old_pairwise = [
        old_llm.QueryResult(relation=rel, jepa_rank=j, llm_rank=m, **shared)
        for rel, (j, m) in zip(relations, ranks)
    ]
    old_three = [
        old_three_way.ComparisonResult(
            relation=rel, loo_jepa_rank=j, v6_jepa_rank=j, llm_rank=m, **shared
        )
        for rel, (j, m) in zip(relations, ranks)
    ]

    assert vs_llm._summarize(new, "jepa_rank") == old_llm._summarize(
        old_pairwise, "jepa_rank"
    )
    assert vs_llm._summarize(new, "llm_rank") == old_three_way._summarize(
        old_three, "llm_rank"
    )


# --------------------------------------------------------------------------- #
# The comparison table: per-arm counts, not one arm's count reused.
# --------------------------------------------------------------------------- #
def _metrics(rows):
    return {
        "mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0,
        "n": sum(row["n"] for row in rows), "per_rel": rows,
    }


def _row(rel, n):
    return {
        "rel": rel, "n": n, "mrr": 0.5, "h1": 0.25, "h3": 0.5, "h10": 0.9,
        "C": 4.0, "chance_mrr": 0.5, "chance_h1": 0.25,
    }


def test_comparison_table_prints_each_arms_own_n(capsys):
    """The failure this phase is most likely to hide.

    The arms run separate pipelines over the same admissions, so a relation's
    `n` differs between them. The old `_print_three_way` printed one `n` column
    taken from the v6 arm, which would silently mislabel both other rows, and
    dropped any relation an arm was missing. Every row now carries its own
    count, and a missing relation prints as `n=0` rather than vanishing.
    """
    vs_fawkes._print_by_relation(
        {
            "fawkes": _metrics([_row("MANAGED_FOR", 5527), _row("CONFIRMS", 125)]),
            "clinical_jepa": _metrics(
                [_row("MANAGED_FOR", 4983), _row("TAKES_MEDICATION", 9530)]
            ),
            "llm": _metrics([_row("MANAGED_FOR", 100)]),
        }
    )
    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[PER-REL] MANAGED_FOR")
        or line.startswith("[PER-REL] TAKES_MEDICATION")
        or line.startswith("[PER-REL] CONFIRMS")
    ]

    managed = [line for line in lines if "MANAGED_FOR" in line]
    assert len(managed) == 3
    assert [line.split()[-5] for line in managed] == ["5527", "4983", "100"]

    # A relation only one arm ranks still shows the other arms, as zero.
    absent = [line for line in lines if "TAKES_MEDICATION" in line]
    assert len(absent) == 3
    assert sum(1 for line in absent if line.split()[-5] == "0") == 2


def test_print_by_relation_tolerates_a_skipped_llm_arm(capsys):
    """`--skip-llm` passes `None` for that arm; the table must still print."""
    vs_fawkes._print_by_relation(
        {
            "fawkes": _metrics([_row("CONFIRMS", 125)]),
            "clinical_jepa": _metrics([_row("CONFIRMS", 59)]),
            "llm": None,
        }
    )
    out = capsys.readouterr().out
    assert "llm" not in out
    assert out.count("[PER-REL] CONFIRMS") == 2


# --------------------------------------------------------------------------- #
# The two vocabularies really do meet: a coverage measurement, not an assumption.
# --------------------------------------------------------------------------- #
@requires_data
def test_clinical_jepa_schema_covers_the_whole_dataset():
    """Aligning the arms on relation name is only meaningful if the relations
    survive the `clinical_jepa` adapter. Measured over all 4,000 records:
    267,952 of 267,952 relation instances map into `EDGE_TYPE_TO_IDX` and every
    node type maps into `NODE_TYPE_TO_IDX`, so no edge is dropped to make the
    comparison work.
    """
    raw, _demographics = _load_graphs(DATA, None)
    total = mapped = 0
    unmapped_relations: Counter = Counter()
    unmapped_types: Counter = Counter()
    for record in raw:
        for node in record.get("nodes", []):
            node_type = str(node.get("type") or "").upper()
            if node_type not in NODE_TYPE_TO_IDX:
                unmapped_types[node_type] += 1
        for edge in adapt_mimic_subkg(record).edges:
            total += 1
            if edge["type"] in EDGE_TYPE_TO_IDX:
                mapped += 1
            else:
                unmapped_relations[edge["type"]] += 1

    assert len(raw) == 4000
    assert not unmapped_types, dict(unmapped_types)
    assert not unmapped_relations, dict(unmapped_relations)
    assert (mapped, total) == (267952, 267952)


# --------------------------------------------------------------------------- #
# Packaging and boundary properties.
# --------------------------------------------------------------------------- #
def test_vs_fawkes_is_the_cross_lineage_bridge():
    """`test_import_boundaries` asserts only `benchmarks` may import both
    lineages. Until now nothing imported both, so that assertion was vacuously
    true. This is what makes it bite."""
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse((BENCHMARKS_SRC / "vs_fawkes.py").read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {"fawkes", "clinical_jepa"} <= roots


@requires_checkpoints
@requires_paper_checkpoint
def test_checkpoint_defaults_point_at_files_that_exist():
    """Plan section 5.2. The old defaults were `ckpts/fawkes_raw_v6/...`,
    `checkpoints/graph_jepa_v6.pt` and `outputs/fawkes_raw_global_v6/test` —
    none of which this repository has ever contained."""
    assert (ROOT / vs_llm.build_arg_parser().get_default("checkpoint")).is_file()
    parser = vs_fawkes.build_arg_parser()
    for option in ("checkpoint", "fawkes_checkpoint"):
        assert (ROOT / parser.get_default(option)).is_file(), option


def test_modules_import_and_parse_args_without_an_api_key():
    """Constructing the CLI must not construct the LLM client.

    `ChatRanker.__init__` imports `openai` and builds a client; `_api_key`
    raises `SystemExit` when no key is configured. Both are reached only inside
    `run`, and only when the LLM arm is enabled — so `--help`, and importing
    either module, works on a machine with no credentials.
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")}
    assert not {"OPENROUTER_API_KEY", "CEREBRAS_API_KEY"} & set(env)
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from benchmarks import vs_fawkes, vs_llm;"
            "vs_llm.build_arg_parser().parse_args([]);"
            "vs_fawkes.build_arg_parser().parse_args(['--skip-llm'])",
        ],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_loo_baseline_architecture_is_gone():
    """Plan §7.1 and §2.3: `LooEncoder` / `LooDistMult` / `LooMLPScorer` and their
    `LOO_*` vocabularies were a third copy of the paper-lineage architecture,
    living inside the modular packages. The v16 checkpoint replaces the arm they
    served, so they are not ported — that is how §7.1 lands here.

    Definitions, not text: `vs_fawkes`'s docstring names all three deliberately,
    and that is the record of why they went.
    """
    assert all(
        hasattr(old_three_way, name)
        for name in ("LooEncoder", "LooDistMult", "LooMLPScorer", "LOO_NUMERIC_DIM")
    ), "old_src is the oracle for what was removed; it must still define them"

    defined: set[str] = set()
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
    assert not {name for name in defined if name.startswith(("Loo", "LOO_"))}, defined
