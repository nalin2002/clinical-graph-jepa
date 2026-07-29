"""Compare the ``fawkes`` paper checkpoint, ``clinical_jepa``, and an LLM.

Merged from ``graph_jepa_v{5,6}/evaluate_loo_v12_jepa_llm.py``, with one
substitution that changes what the module is.

**The baseline arm is the v16 paper checkpoint, not the v12 LOO checkpoint.**
The old script's first arm was a Hugging Face LOO baseline
(``fawkes_jepa_loo_eval_v12_260615.pt``) reimplemented locally as ``LooEncoder``
/ ``LooDistMult`` / ``LooMLPScorer`` — plan §2.3's third copy of the
paper-lineage architecture, living inside the modular packages. That checkpoint
is not present in this repository and is not being obtained, so the arm could
never run. ``models/paper_v16/`` *is* present and its behaviour is pinned
exactly by Phase 0, so the baseline arm is now ``fawkes`` itself and the
comparison has a numeric gate it never had before. The three classes and their
``LOO_*`` vocabularies are deleted; plan §7.1 lands by this route rather than by
loading v12 into ``fawkes.model.Encoder``.

The v16 checkpoint was **not** loaded into ``LooEncoder`` and could not be:
``LOO_NUMERIC_DIM`` was 6 where v16 records ``numeric_dim: 774`` (6 + 768 note
dimensions). Nothing here shows the two architectures are equivalent — see
``docs/LINEAGE.md``.

Reading the output — the arms are not paired
--------------------------------------------
The old script fed both model arms from one ``PatientGraphDataset``, because
``LooEncoder`` was written to eat those tensors, and so it could rank all three
arms on one sampled query and print a single ``n`` column. ``fawkes`` cannot do
that: it has its own ``to_data`` with a different feature layout, its own node
and relation vocabularies, and its own edge pruning. So each arm runs its own
pipeline and the table aligns on **relation name only**.

Consequently every row carries **its own** ``n``, ``C`` and ``chance`` — a
single shared ``n`` would be wrong here. Concretely, on the 400-graph test split
the two model arms cover different edge populations (measured, this dataset):

* ``fawkes`` keeps a graph when ``num_nodes >= 3 and edge_index.size(1) >= 4``
  and prunes LLM edges with no evidence; ``clinical_jepa`` keeps every edge
  whose relation is in ``EDGE_TYPE_TO_IDX`` and then filters candidates through
  its typed schema and LLM-confidence masks.
* ``clinical_jepa`` additionally evaluates ``TAKES_MEDICATION``, which ``fawkes``
  never ranks: its clinical-artifact filter demotes some medication edges out of
  the positive set, which leaves them behind as distractors and makes the
  remaining ones rankable.

A relation showing ``n=0`` for ``fawkes`` is a **protocol artifact of star
topology under filtered ranking, not a model deficiency.** ``loo_evaluate``
builds candidates as "all nodes of the true tail's type", then removes the other
true tails of ``(u, rel)`` — the standard filtered setting — and skips the query
when fewer than two candidates survive. Every patient-centric relation therefore
collapses: the patient is joined to *every* node of the target type, so the
filter empties the pool. Measured on 40 admissions, mean pool size before and
after the filter::

    TAKES_MEDICATION      30.0 -> 1.0     (0 ranked, 999 dropped)
    HAS_DIAGNOSIS         17.6 -> 1.0     (0 ranked, 595 dropped)
    UNDERWENT_PROCEDURE    2.4 -> 1.0     (0 ranked,  43 dropped)
    MANAGED_FOR           15.6 -> 15.1    (291 ranked, 0 dropped)
    COMPLICATED_BY        18.1 -> 16.5    ( 39 ranked, 0 dropped)

"Which of these 30 medications does this patient take?" is not a well-posed
ranking question when the answer is "all of them". Relations *between* clinical
entities keep their distractors and survive.

The practical consequence: the two model arms construct candidate sets
differently, so **their MRR is not directly comparable even on the relations
both rank.** Read any cross-arm comparison against each arm's own ``C`` and
``chance_mrr``, which is why both columns are printed per row.

Both arms do share their *admissions*: the seeded test split is computed once,
from ``fawkes``' filter, and the same records feed ``clinical_jepa``. The
LLM arm is a bounded sample of ``clinical_jepa``'s queries with truncated
candidate lists (``--max-candidates``), so its ``C`` is not the other arms' and
its MRR is only interpretable against its own ``chance`` column and against the
``clinical_jepa (sampled)`` line printed beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from clinical_jepa.encoders import build_checkpoint_encoder
from clinical_jepa.evaluate import leave_one_out_recovery
from clinical_jepa.graph.builders import adapt_mimic_subkg
from clinical_jepa.graph.tensors import PatientGraphDataset
from clinical_jepa.schema import IDX_TO_EDGE_TYPE
from clinical_jepa.train.loop import load_model_checkpoint
from fawkes.config import Config
from fawkes.data import to_data
from fawkes.evaluate import _load_graphs, loo_evaluate
from fawkes.model import DistMult, Encoder

from .llm_ranker import ChatRanker, _api_base, _api_key, _node_label
from .vs_llm import (
    QueryResult,
    _print_summary,
    _rank_from_order,
    _summarize,
    build_prompt,
    collect_recovery_queries,
    jepa_rank_query,
    parse_ranking,
)

DEFAULT_DATA = (
    "data/fawkes-training-graph-embedded-260615/"
    "fawkes_training_graph_full_embedded_260615.jsonl"
)
DEFAULT_FAWKES_CHECKPOINT = (
    "models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt"
)
DEFAULT_CHECKPOINT = "models/v6_with_note/graph_jepa_v6.pt"


def paper_test_split(raw: list, demographics: dict, cfg: Config) -> tuple[list, list[int]]:
    """The trainer's seeded TEST split — ``fawkes.train.main``, the split block.

    Returns the converted graphs and the indices of the raw records they came
    from, so the ``clinical_jepa`` arm can be run over the same admissions. The
    ``>= 4`` edge filter is the trainer's, not ``fawkes.evaluate.run``'s ``>= 2``;
    the two select different populations, and this one is what
    ``baseline/paper_loo_testsplit.json`` was recorded under.
    """
    items: list = []
    records: list[int] = []
    for index, graph in enumerate(raw):
        data = to_data(graph, demographics, cfg)
        if data.num_nodes >= 3 and data.edge_index.size(1) >= 4:
            items.append(data)
            records.append(index)
    order = np.random.RandomState(cfg.seed).permutation(len(items))
    selected = order[: int(cfg.test_frac * len(items))]
    return [items[i] for i in selected], [records[i] for i in selected]


def fawkes_arm(
    checkpoint_path: str,
    raw: list,
    demographics: dict,
    device: torch.device,
    *,
    cap: int,
) -> tuple[dict, list[int], Config]:
    """The v16 baseline: ``fawkes``' own pipeline, end to end.

    Reproduces ``baseline/paper_loo_testsplit.json`` exactly (Phase 0 measured a
    delta of ``0.000e+00``), which is what makes this arm a gate rather than a
    number nobody can check.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config.from_checkpoint(checkpoint["config"])
    encoder = Encoder(cfg).to(device)
    scorer = DistMult(cfg).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
    graphs, records = paper_test_split(raw, demographics, cfg)
    return loo_evaluate(encoder, scorer, graphs, device, cfg, cap=cap), records, cfg


def clinical_jepa_pipeline(raw: list, records: list[int], encoder, cfg, *, source: str):
    """The second pipeline over the same admissions: JSONL records -> PyG tensors.

    ``adapt_mimic_subkg`` is what ``JsonlGraphBuilder`` applies per line; it is
    called directly here because the record selection comes from ``fawkes``'
    seeded split rather than from a whole file.
    """
    graphs = [
        adapt_mimic_subkg(raw[index], source_path=f"{source}:{index + 1}")
        for index in records
    ]
    dataset = PatientGraphDataset(
        graphs,
        encoder,
        use_note_embeddings=cfg.model.use_note_embeddings,
        note_embedding_dim=cfg.model.note_embedding_dim,
        note_ground_by=cfg.model.note_ground_by,
    )
    return graphs, [dataset[idx] for idx in range(len(dataset))]


def _print_by_relation(arms: dict[str, dict]) -> None:
    """One row per (relation, arm). Every row carries that arm's own counts.

    The old ``_print_three_way`` printed a single ``n`` column taken from the v6
    arm and dropped any relation an arm was missing. That was safe when one
    pipeline fed every arm. It is not safe now — see the module docstring.
    """
    by_arm = {
        arm: {row["rel"]: row for row in metrics["per_rel"]}
        for arm, metrics in arms.items()
        if metrics is not None
    }
    relations = sorted(
        {rel for rows in by_arm.values() for rel in rows},
        key=lambda rel: -max(rows.get(rel, {}).get("n", 0) for rows in by_arm.values()),
    )
    print("[PER-REL] n/C/chance are per arm; the arms rank different edge populations")
    print(
        f"[PER-REL] {'relation':<22} {'arm':<14} {'n':>6} {'C':>6} "
        f"{'chance':>7} {'MRR':>7} {'H@1':>7}"
    )
    for relation in relations:
        for arm, rows in by_arm.items():
            row = rows.get(relation)
            if row is None:
                print(
                    f"[PER-REL] {relation:<22} {arm:<14} {0:>6} {'-':>6} "
                    f"{'-':>7} {'-':>7} {'-':>7}"
                )
                continue
            print(
                f"[PER-REL] {relation:<22} {arm:<14} {row['n']:>6} {row['C']:>6.1f} "
                f"{row['chance_mrr']:>7.3f} {row['mrr']:>7.3f} {row['h1']:>7.3f}"
            )


def run(args) -> dict:
    device = torch.device(args.device)

    raw, demographics = _load_graphs(Path(args.data), None)
    fawkes_metrics, records, fawkes_cfg = fawkes_arm(
        args.fawkes_checkpoint,
        raw,
        demographics,
        device,
        cap=args.cap,
    )

    model, cfg = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    encoder = build_checkpoint_encoder(cfg, args.encoder_cache)
    graphs, data_list = clinical_jepa_pipeline(
        raw,
        records,
        encoder,
        cfg,
        source=args.data,
    )
    jepa_metrics = leave_one_out_recovery(
        model,
        data_list,
        cfg,
        device,
        cap=args.cap,
        candidate_mode=args.candidate_mode,
    )

    print(
        f"[POP] records={len(raw)} test_split={len(records)} "
        f"seed={fawkes_cfg.seed} test_frac={fawkes_cfg.test_frac} "
        f"(fawkes filter: num_nodes>=3, edges>=4)"
    )
    print(
        f"[POP] fawkes edges={fawkes_metrics['n']} "
        f"clinical_jepa edges={jepa_metrics['n']} "
        "— different pipelines, so different populations"
    )

    queries: list = []
    results: list[QueryResult] = []
    counts: Counter = Counter()
    llm_model = ""
    llm_metrics = None
    sampled_metrics = None
    if not args.skip_llm:
        queries = collect_recovery_queries(
            data_list,
            cfg,
            samples_per_relation=args.samples_per_relation,
            candidate_mode=args.candidate_mode,
            max_candidates=args.max_candidates,
            seed=args.seed,
            allow_duplicate_triples=args.allow_duplicate_triples,
        )
        if args.max_queries is not None:
            queries = queries[: args.max_queries]
        if not queries:
            raise ValueError("no eligible LOO-compatible recovery queries found")

        counts = Counter(
            IDX_TO_EDGE_TYPE.get(q.relation, f"rel{q.relation}") for q in queries
        )
        print(
            f"[SAMPLE] queries={len(queries)} candidate_mode={args.candidate_mode} "
            f"samples_per_relation={args.samples_per_relation} relations={dict(counts)}"
        )

        llm_model = args.llm_model or os.environ.get(f"{args.provider.upper()}_MODEL", "")
        if not llm_model:
            raise SystemExit(
                f"Pass --llm-model or set {args.provider.upper()}_MODEL in the environment."
            )
        ranker = ChatRanker(
            provider=args.provider,
            model=llm_model,
            api_key=_api_key(args.provider),
            base_url=args.api_base or _api_base(args.provider),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=(
                None if args.reasoning_effort == "none" else args.reasoning_effort
            ),
            reasoning_format=(
                None if args.reasoning_format == "none" else args.reasoning_format
            ),
            retries=args.retries,
            sleep=args.retry_sleep,
        )

        records_path = Path(args.records_output) if args.records_output else None
        if records_path:
            records_path.parent.mkdir(parents=True, exist_ok=True)
            records_path.write_text("", encoding="utf-8")

        for idx, query in enumerate(queries, start=1):
            graph = graphs[query.graph_index]
            data = data_list[query.graph_index]
            relation = IDX_TO_EDGE_TYPE.get(query.relation, f"rel{query.relation}")
            prompt = build_prompt(
                graph,
                data,
                cfg,
                query,
                context_mode=args.context,
                max_context_edges=args.max_context_edges,
            )
            response, usage = ranker.rank(prompt)
            order, parse_ok, parse_complete = parse_ranking(
                response,
                len(query.candidates),
            )
            true_position = query.candidates.index(query.target)
            llm_rank = (
                _rank_from_order(order, true_position)
                if parse_ok
                else len(query.candidates)
            )
            result = QueryResult(
                graph_index=query.graph_index,
                edge_index=query.edge_index,
                relation=relation,
                source=_node_label(graph, query.source),
                target=_node_label(graph, query.target),
                candidates=[
                    _node_label(graph, candidate)
                    for candidate in query.candidates
                ],
                # The paired reading: the same query, scored by clinical_jepa.
                # Without it the LLM row cannot be compared to anything, because
                # its candidate lists are truncated and its edges are a sample.
                jepa_rank=jepa_rank_query(model, data, query, cfg, device),
                llm_rank=llm_rank,
                llm_parse_ok=parse_ok,
                llm_parse_complete=parse_complete,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                finish_reason=str(usage.get("finish_reason", "")),
                llm_response=response,
            )
            results.append(result)
            if records_path:
                with records_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
            if idx % args.progress_every == 0 or idx == len(queries):
                print(f"[PROGRESS] {idx}/{len(queries)} queries complete", flush=True)
            if args.request_sleep > 0:
                time.sleep(args.request_sleep)

        llm_metrics = _summarize(results, "llm_rank")
        sampled_metrics = _summarize(results, "jepa_rank")

    print("[RESULTS]")
    _print_summary("fawkes(v16)", fawkes_metrics)
    _print_summary("clinical_jepa", jepa_metrics)
    if llm_metrics is not None:
        parse_failures = sum(1 for result in results if not result.llm_parse_ok)
        incomplete = sum(1 for result in results if not result.llm_parse_complete)
        token_usage = {
            "prompt_tokens": sum(result.prompt_tokens for result in results),
            "completion_tokens": sum(result.completion_tokens for result in results),
            "reasoning_tokens": sum(result.reasoning_tokens for result in results),
            "total_tokens": sum(result.total_tokens for result in results),
        }
        finish_reasons = Counter(result.finish_reason for result in results)
        _print_summary("clinical_jepa (sampled)", sampled_metrics)
        _print_summary("llm", llm_metrics)
        print(
            f"[LLM] parse_failures={parse_failures}/{len(results)} "
            f"incomplete_rankings={incomplete}/{len(results)}"
        )
        print(
            "[LLM] tokens "
            f"prompt={token_usage['prompt_tokens']} "
            f"completion={token_usage['completion_tokens']} "
            f"reasoning={token_usage['reasoning_tokens']} "
            f"total={token_usage['total_tokens']}"
        )
        print(f"[LLM] finish_reasons={dict(finish_reasons)}")

    _print_by_relation(
        {"fawkes": fawkes_metrics, "clinical_jepa": jepa_metrics, "llm": llm_metrics}
    )

    payload = {
        "config": {
            "data": args.data,
            "fawkes_checkpoint": args.fawkes_checkpoint,
            "checkpoint": args.checkpoint,
            "cap": args.cap,
            "candidate_mode": args.candidate_mode,
            "provider": args.provider,
            "llm_model": llm_model,
            "context": args.context,
            "samples_per_relation": args.samples_per_relation,
            "max_candidates": args.max_candidates,
            "max_context_edges": args.max_context_edges,
            "seed": args.seed,
            "allow_duplicate_triples": args.allow_duplicate_triples,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_format": args.reasoning_format,
            "skip_llm": args.skip_llm,
        },
        "population": {
            "records": len(raw),
            "test_split_graphs": len(records),
            "split_seed": fawkes_cfg.seed,
            "test_frac": fawkes_cfg.test_frac,
            "fawkes_edges": fawkes_metrics["n"],
            "clinical_jepa_edges": jepa_metrics["n"],
            "llm_queries": len(queries),
        },
        "sample_counts": dict(counts),
        "fawkes": fawkes_metrics,
        "clinical_jepa": jepa_metrics,
        "records": [asdict(result) for result in results],
    }
    if llm_metrics is not None:
        payload["clinical_jepa_sampled"] = sampled_metrics
        payload["llm"] = llm_metrics
        payload["llm_parse_failures"] = parse_failures
        payload["llm_incomplete_rankings"] = incomplete
        payload["llm_token_usage"] = token_usage
        payload["llm_finish_reasons"] = dict(finish_reasons)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the fawkes paper checkpoint, clinical_jepa, and an LLM on "
            "leave-one-out edge recovery over one clinical-graph JSONL file."
        )
    )
    # No add_data_args: both arms must read the same JSONL, and fawkes' pipeline
    # cannot consume the synthetic/mimic/aci-bench sources that parser offers.
    parser.add_argument("--data", default=DEFAULT_DATA, help="Clinical graph JSONL")
    parser.add_argument("--fawkes-checkpoint", default=DEFAULT_FAWKES_CHECKPOINT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-cache", default=".cache/clinical_jepa/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=40000)
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "same-type"],
        default="same-type",
        help="same-type matches how fawkes builds its candidate set",
    )
    parser.add_argument("--samples-per-relation", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-duplicate-triples",
        action="store_true",
        help="Allow exact duplicate triples that can leak the hidden edge",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only compare the two model arms; skip all LLM calls.",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "cerebras"],
        default="openrouter",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Provider model id. Or set OPENROUTER_MODEL/CEREBRAS_MODEL.",
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="low",
    )
    parser.add_argument(
        "--reasoning-format",
        choices=["none", "parsed", "raw", "hidden"],
        default="hidden",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument(
        "--context",
        choices=["none", "source", "full"],
        default="full",
    )
    parser.add_argument("--max-context-edges", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", default=None, help="Final JSON summary path")
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL path written incrementally after each LLM call",
    )
    return parser


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
