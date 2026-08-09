"""Aggregate the paired v22/v23 note-injection checkpoints from Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch
from huggingface_hub import hf_hub_download


SEEDS = range(42, 52)
ARMS = {
    "v22-mean": "kushagrayadv/fawkes-v22-patch-mlp-sp42-s{seed}",
    "v23-uniform": "nalin9/fawkes-v23-uniform-note-sp42-s{seed}",
    "v23-attention": "nalin9/fawkes-v23-attention-note-sp42-s{seed}",
}
LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BATCH_METRICS = ("auc", "ap", "mrr")
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP"}
T_CRITICAL_95 = {10: 2.262, 9: 2.306, 8: 2.365, 7: 2.447, 6: 2.571,
                 5: 2.776, 4: 3.182, 3: 4.303, 2: 12.706}


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs():
    runs = {}
    for arm, template in ARMS.items():
        for seed in SEEDS:
            repo = template.format(seed=seed)
            try:
                path = hf_hub_download(repo, "fawkes_entity_note.pt")
            except Exception as exc:
                print(f"[MISSING] {arm} seed={seed}: {type(exc).__name__}")
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            runs[(arm, seed)] = {
                "loo": checkpoint["recovery_test_loo"],
                "batch": checkpoint["recovery_test_batchmask"],
                "config": checkpoint.get("config", {}),
            }
    return runs


def summarize(runs, block, metrics):
    print(f"\n{block.upper()} mean ± sample sd")
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {}
        cells = []
        for metric in metrics:
            values = [runs[(arm, seed)][block][metric] for seed in SEEDS
                      if (arm, seed) in runs]
            if values:
                mean, sd = mean_sd(values)
                summaries[arm][metric] = {"mean": mean, "sd": sd, "n": len(values)}
                cells.append(f"{LABELS[metric]}={mean:.4f}±{sd:.4f}")
        print(f"{arm:<15} {'  '.join(cells)}")
    return summaries


def paired(runs, new_arm, baseline_arm, block, metrics):
    result = {}
    print(f"\nPAIRED {new_arm} - {baseline_arm} ({block})")
    for metric in metrics:
        diffs = [runs[(new_arm, seed)][block][metric] - runs[(baseline_arm, seed)][block][metric]
                 for seed in SEEDS if (new_arm, seed) in runs and (baseline_arm, seed) in runs]
        if not diffs:
            continue
        mean, sd = mean_sd(diffs)
        n = len(diffs)
        t_value = T_CRITICAL_95.get(n)
        half = t_value * sd / n ** 0.5 if t_value and n > 1 else float("nan")
        wins = sum(value > 0 for value in diffs)
        result[metric] = {"mean_delta": mean, "sd_delta": sd, "ci95": [mean - half, mean + half],
                          "wins": wins, "n": n, "per_seed": diffs}
        print(f"{LABELS[metric]:<5} Δ={mean:+.4f}±{sd:.4f}  95% CI [{mean-half:+.4f}, {mean+half:+.4f}]  wins={wins}/{n}")
    return result


def main():
    runs = load_runs()
    result = {
        "loo": summarize(runs, "loo", LOO_METRICS),
        "batch": summarize(runs, "batch", BATCH_METRICS),
        "paired": {},
    }
    for new_arm, baseline_arm in (
            ("v23-uniform", "v22-mean"),
            ("v23-attention", "v22-mean"),
            ("v23-attention", "v23-uniform")):
        key = f"{new_arm}_minus_{baseline_arm}"
        result["paired"][key] = {
            "loo": paired(runs, new_arm, baseline_arm, "loo", LOO_METRICS),
            "batch": paired(runs, new_arm, baseline_arm, "batch", BATCH_METRICS),
        }
    output = Path("results/v23-note-injection-summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[DONE] {output}")


if __name__ == "__main__":
    main()
