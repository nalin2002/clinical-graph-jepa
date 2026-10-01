"""Aggregate Clinical-JEPA note/no-note HF seed sweeps.

Expected repo names come from ``submit-clinical-jepa-seed-variance.sh``:

    <owner>/clinical-jepa-{note,no-note}-fawkes-sp42-s{42..51}

Each repo stores ``shared_queries_eval.json`` from ``benchmarks.shared_queries``.
The comparison is paired by seed: same Fawkes split, same 8,283-query manifest,
different Clinical-JEPA variant.
"""

from __future__ import annotations

import math
import os
import statistics

from huggingface_hub import HfApi, hf_hub_download


SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
ARMS = {
    "note": "clinical-jepa-note-fawkes",
    "no-note": "clinical-jepa-no-note-fawkes",
}
METRICS = ("mrr", "hits1", "hits3", "hits10")
LABELS = {
    "mrr": "MRR",
    "hits1": "H@1",
    "hits3": "H@3",
    "hits10": "H@10",
}
T95 = 2.262  # t(9, .975), for 10 paired seeds.
CACHE = "data/clinical_jepa_seed_variance"


def owner() -> str:
    explicit = os.environ.get("CLINICAL_JEPA_HF_USER") or os.environ.get("HF_USER")
    if explicit:
        return explicit
    return HfApi().whoami()["name"]


def mean_sd(values):
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan"), 0
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return statistics.mean(clean), sd, len(clean)


def cell(value: float) -> str:
    return f"{'nan':>11}" if value is None or math.isnan(value) else f"{value:>11.4f}"


def main() -> None:
    user = owner()
    runs = {}
    fawkes = None
    for arm, stem in ARMS.items():
        for seed in SEEDS:
            label = f"sp{SPLIT}-s{seed}"
            repo = f"{user}/{stem}-{label}"
            try:
                path = hf_hub_download(
                    repo,
                    "shared_queries_eval.json",
                    local_dir=f"{CACHE}/{arm}-{label}",
                )
            except Exception as exc:
                print(f"{arm:<8} {label} MISSING ({type(exc).__name__})")
                continue
            import json

            payload = json.loads(open(path, encoding="utf-8").read())
            runs[(arm, seed)] = payload["clinical_jepa"]
            fawkes = fawkes or payload.get("fawkes")
            coverage = payload.get("coverage", {})
            if coverage.get("clinical_jepa_fatal_skips", 0):
                print(f"WARNING {arm} {label}: coverage={coverage}")

    paired = [seed for seed in SEEDS if ("note", seed) in runs and ("no-note", seed) in runs]
    if not paired:
        raise SystemExit("no seed has both note and no-note runs")
    if len(paired) < len(SEEDS):
        print(f"\nWARNING: only {len(paired)}/{len(SEEDS)} paired seeds: {paired}")

    print("\nHEADLINE")
    print(f"{'arm':<9}" + "".join(f"{LABELS[m]:>11}" for m in METRICS) + f"{'n':>8}")
    if fawkes:
        print(
            f"{'fawkes':<9}"
            + "".join(cell(fawkes[m]) for m in METRICS)
            + f"{fawkes['n']:>8}"
        )
    for arm in ARMS:
        values = {metric: [runs[(arm, seed)][metric] for seed in paired] for metric in METRICS}
        n_values = {runs[(arm, seed)]["n"] for seed in paired}
        print(
            f"{arm:<9}"
            + "".join(cell(mean_sd(values[metric])[0]) for metric in METRICS)
            + f"{min(n_values):>8}"
        )
        print(f"{'sd':<9}" + "".join(cell(mean_sd(values[metric])[1]) for metric in METRICS))

    print(f"\nPAIRED DELTA: no-note - note, n={len(paired)}")
    print(f"{'seed':<7}" + "".join(f"{LABELS[m] + ' delta':>14}" for m in METRICS))
    for seed in paired:
        note = runs[("note", seed)]
        no_note = runs[("no-note", seed)]
        print(
            f"{seed:<7}"
            + "".join(f"{no_note[metric] - note[metric]:>+14.4f}" for metric in METRICS)
        )

    print("\npaired mean differences:")
    for metric in METRICS:
        diffs = [
            runs[("no-note", seed)][metric] - runs[("note", seed)][metric]
            for seed in paired
        ]
        mean, sd, n = mean_sd(diffs)
        if n < 2:
            continue
        half = T95 * sd / math.sqrt(n)
        lo, hi = mean - half, mean + half
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
        wins = sum(1 for diff in diffs if diff > 0)
        print(
            f"  {LABELS[metric]:<5} {mean:+.4f} +/- {sd:.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"{verdict:<26} no-note ahead at {wins}/{n}"
        )


if __name__ == "__main__":
    main()
