"""Re-score auc_nonobvious on every cached checkpoint, both ways.

The metric shipped in `_classification_metrics` filtered the positives to
non-PATIENT-anchored edges but scored them against the *unfiltered* negative
pool. v21 established the defect and measured it on three seeds; this script
runs the correction across all thirty v20/v21/v22 checkpoints so the column can
be quoted.

Both computations run in the same pass over the same forward scores, so the
`shipped` column here is directly comparable to the value stored inside each
checkpoint — which is what establishes the reproduction before the `matched`
column is read.

Everything runs on CPU. `freeze_eval` seeds the held/negative draw per graph id,
but a `torch.Generator` on CPU does not reproduce a CUDA generator's stream, so
absolute values shift slightly against the fleet's stored numbers. Every arm
here is drawn identically, so the paired comparisons are unaffected — the same
caveat, and the same reasoning, as v21 §4.
"""
import math
import statistics

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from fawkes.config import Config
from fawkes.model import Encoder, build_scorer
from fawkes.steps import readout_step
from fawkes.train import prepare_data

DATA = "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"
PAPER_CKPT = "models/fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt"
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
T95 = 2.262   # t(9, .975)

# name -> (decoder, local cache dir stem). The v20 checkpoints are cached under two
# names: data/fawkes_v20/ from the variance sweep, and data/fawkes_v21/distmult-* as
# v21's paired baseline. They are the same ten files.
ARMS = {
    "v20": ("distmult", "data/fawkes_v20/"),
    "v21": ("mlp",      "data/fawkes_v21/mlp-"),
    "v22": ("mlp",      "data/fawkes_v22/v22-"),
}
COMPARISONS = [("v21", "v20", "readout head"),
               ("v22", "v21", "mask strategy"),
               ("v22", "v20", "the full stack")]


@torch.no_grad()
def score_both_ways(encoder, scorer, loader, device, cfg):
    """One forward pass; AUC/AP plus auc_nonobvious computed shipped and matched."""
    encoder.eval()
    scorer.eval()
    pos, neg, nonpat = [], [], []
    for batch in loader:
        gen = torch.Generator(device=device).manual_seed(int(batch.gid[0]) & 0x7FFFFFFFFFFFFFFF) if cfg.freeze_eval else None
        step_out = readout_step(encoder, scorer, batch, device, False, cfg, gen=gen)
        if step_out is None:
            continue
        _, pos_scores, neg_scores, nonpatient, _ = step_out
        pos.append(pos_scores)
        neg.append(neg_scores)
        nonpat.append(nonpatient)

    positives = torch.cat(pos).cpu().numpy()
    negatives = torch.cat(neg).cpu().numpy()
    mask = torch.cat(nonpat).cpu().numpy()
    labels = np.concatenate([np.ones_like(positives), np.zeros_like(negatives)])
    scores = np.concatenate([positives, negatives])

    def auc_of(p, n):
        return roc_auc_score(np.concatenate([np.ones_like(p), np.zeros_like(n)]),
                             np.concatenate([p, n]))

    return {"auc": roc_auc_score(labels, scores),
            "ap": average_precision_score(labels, scores),
            "shipped": auc_of(positives[mask], negatives),           # the defect
            "matched": auc_of(positives[mask], negatives[mask]),     # the fix
            "n_pos": len(positives), "n_nonpat": int(mask.sum())}


def mean_sd(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


cfg = Config(data_path=DATA, data_split_seed=SPLIT)
_train, _val, test_pairs = prepare_data(cfg)
test_graphs = [d for d, _ in test_pairs]
device = torch.device("cpu")
loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
print(f"TEST graphs={len(test_graphs)} split={SPLIT} device=cpu")

runs = {}
for arm, (decoder, stem) in ARMS.items():
    arm_cfg = Config(data_path=DATA, data_split_seed=SPLIT, decoder=decoder)
    for seed in SEEDS:
        path = f"{stem}sp{SPLIT}-s{seed}/fawkes_entity_note.pt"
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print(f"{arm:<4} s{seed}  MISSING {path}")
            continue
        encoder = Encoder(arm_cfg).to(device)
        scorer = build_scorer(arm_cfg).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
        scorer.load_state_dict(checkpoint["scorer"])
        metrics = score_both_ways(encoder, scorer, loader, device, arm_cfg)
        metrics["stored"] = checkpoint.get("recovery_test_batchmask", {}).get("auc_nonobvious")
        runs[(arm, seed)] = metrics
        print(f"{arm:<4} s{seed}  shipped={metrics['shipped']:.4f} (stored {metrics['stored']:.4f})  "
              f"matched={metrics['matched']:.4f}  auc={metrics['auc']:.4f}")

present = [a for a in ARMS if any((a, s) in runs for s in SEEDS)]
if not present:
    raise SystemExit("no checkpoints found — nothing to report")

print(f"\n{'=' * 78}\nPER ARM — mean +/- sd over seeds")
print(f"\n{'arm':<6}{'AUC':>11}{'AP':>11}{'nonobv shipped':>17}{'nonobv matched':>17}{'n':>5}")
for arm in present:
    cells = "".join(f"{mean_sd([runs[(arm, s)][k] for s in SEEDS if (arm, s) in runs])[0]:>11.4f}"
                    for k in ("auc", "ap"))
    ship = mean_sd([runs[(arm, s)]["shipped"] for s in SEEDS if (arm, s) in runs])
    match = mean_sd([runs[(arm, s)]["matched"] for s in SEEDS if (arm, s) in runs])
    print(f"{arm:<6}{cells}{ship[0]:>11.4f}+-{ship[1]:.4f}{match[0]:>11.4f}+-{match[1]:.4f}{ship[2]:>5}")

print(f"\n{'=' * 78}\nPAIRED DIFFERENCES — does the correction change the sign?")
for new, base, what in COMPARISONS:
    seeds = [s for s in SEEDS if (new, s) in runs and (base, s) in runs]
    if len(seeds) < 2:
        continue
    print(f"\n  {new} - {base}  ({what})   paired, n={len(seeds)}")
    for key in ("shipped", "matched"):
        diffs = [runs[(new, s)][key] - runs[(base, s)][key] for s in seeds]
        mean, sd, n = mean_sd(diffs)
        half = T95 * sd / math.sqrt(n)
        lo, hi = mean - half, mean + half
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
        wins = sum(1 for d in diffs if d > 0)
        print(f"    nonobv {key:<8} {mean:+.4f} +- {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  {verdict:<26} ahead at {wins}/{n}")

print(f"\n{'=' * 78}\nTHE RELEASED CHECKPOINT — the published auc_nonobvious of 0.800028")
released_cfg = Config(data_path=DATA, data_split_seed=SPLIT, decoder="distmult")
released = torch.load(PAPER_CKPT, map_location="cpu", weights_only=False)
encoder, scorer = Encoder(released_cfg).to(device), build_scorer(released_cfg).to(device)
encoder.load_state_dict(released["encoder"])
scorer.load_state_dict(released["scorer"])
released_metrics = score_both_ways(encoder, scorer, loader, device, released_cfg)
published = released["recovery_test_batchmask"]["auc_nonobvious"]
print(f"  published (in the checkpoint) {published:.6f}")
print(f"  shipped   (recomputed here)   {released_metrics['shipped']:.6f}   "
      f"delta {released_metrics['shipped'] - published:+.6f}  <- the reproduction check")
print(f"  matched   (corrected)         {released_metrics['matched']:.6f}   "
      f"delta {released_metrics['matched'] - published:+.6f}  <- what Table 1 should say")

print(f"\n{'=' * 78}")
sample = runs[(present[0], SEEDS[0])]
print(f"Population: {sample['n_nonpat']} non-patient positives of {sample['n_pos']} held edges "
      f"({sample['n_nonpat'] / sample['n_pos']:.1%}); the shipped metric scored those "
      f"{sample['n_nonpat']} against all {sample['n_pos']} negatives, of which "
      f"{1 - sample['n_nonpat'] / sample['n_pos']:.1%} belong to patient-anchored edges.")
print("The `shipped` column is computed here, not read from the checkpoints; compare it")
print("with the stored value printed per run above to confirm the reproduction is faithful")
print("before reading the `matched` column.")
