"""v21 (MLP readout) against v20 (DistMult), paired seed by seed.

The two arms share DATA_SPLIT_SEED=42, so every run is scored on the same 8,283 LOO
edges and the same 3,000 batch-mask queries. That makes the per-seed difference a
paired measurement, which is a good deal more sensitive than comparing the two means.
"""
import math
import statistics
import torch
from huggingface_hub import hf_hub_download

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
ARMS = {"distmult": "fawkes-v20-variance", "mlp": "fawkes-v21-mlp"}
CACHE = "/Users/kushagrayadav/Code/clinical-graph-jepa/data/fawkes_v21"
LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "auc_nonobvious", "mrr")
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP", "auc_nonobvious": "AUC_nonobv"}
T95 = 2.262   # t(9, .975); both arms are 10 seeds

runs = {}
for arm, repo_stem in ARMS.items():
    for seed in SEEDS:
        label = f"sp{SPLIT}-s{seed}"
        try:
            path = hf_hub_download(f"kushagrayadv/{repo_stem}-{label}",
                                   "fawkes_entity_note.pt",
                                   local_dir=f"{CACHE}/{arm}-{label}")
        except Exception as exc:
            print(f"{arm:<9} {label}  MISSING ({type(exc).__name__}) — check `hf jobs logs`")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        runs[(arm, seed)] = {"loo": ck["recovery_test_loo"],
                             "bm": ck.get("recovery_test_batchmask")}

paired = [s for s in SEEDS if ("distmult", s) in runs and ("mlp", s) in runs]
if not paired:
    raise SystemExit("no seed has both arms — nothing to compare")
if len(paired) < len(SEEDS):
    print(f"\nWARNING: only {len(paired)}/{len(SEEDS)} seeds have both arms: {paired}")


def mean_sd(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def compare(title, block, metrics):
    print(f"\n{title}")
    print(f"{'seed':<7}" + "".join(f"{LABELS[m] + ' Δ':>14}" for m in metrics))
    for seed in paired:
        a, b = runs[("distmult", seed)][block], runs[("mlp", seed)][block]
        if a is None or b is None:
            print(f"{seed:<7}  (block missing)")
            continue
        print(f"{seed:<7}" + "".join(f"{b[m] - a[m]:>+14.4f}" for m in metrics))

    print(f"\n{'':<7}" + "".join(f"{LABELS[m]:>14}" for m in metrics))
    for name, arm in (("DistMult", "distmult"), ("MLP", "mlp")):
        vals = {m: [runs[(arm, s)][block][m] for s in paired if runs[(arm, s)][block]] for m in metrics}
        mean = "".join(f"{mean_sd(vals[m])[0]:>14.4f}" for m in metrics)
        sd = "".join(f"{'±' + format(mean_sd(vals[m])[1], '.4f'):>14}" for m in metrics)
        print(f"{name:<7}{mean}\n{'':<7}{sd}")

    print(f"\npaired difference (MLP − DistMult), n={len(paired)}:")
    for m in metrics:
        diffs = [runs[("mlp", s)][block][m] - runs[("distmult", s)][block][m]
                 for s in paired if runs[("mlp", s)][block] and runs[("distmult", s)][block]]
        mean, sd, n = mean_sd(diffs)
        if n < 2:
            continue
        half = T95 * sd / math.sqrt(n)
        lo, hi = mean - half, mean + half
        # A 95% CI on the paired mean that excludes zero is exactly a two-sided
        # paired t-test at .05 — computed from the t critical value so the script
        # needs no scipy.
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
        wins = sum(1 for d in diffs if d > 0)
        print(f"  {LABELS[m]:<11} {mean:+.4f} ± {sd:.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {verdict:<26} MLP ahead at {wins}/{n} seeds")


compare("LEAVE-ONE-OUT — per-seed delta, MLP minus DistMult", "loo", LOO_METRICS)
if all(runs[(a, s)]["bm"] for a in ARMS for s in paired):
    compare("BATCH-MASK — per-seed delta, MLP minus DistMult. Not comparable with LOO above.",
            "bm", BM_METRICS)
else:
    print("\nbatch-mask block missing from at least one run — AUC comparison skipped")

print("\nReminder: DECODER is not recorded in checkpoint_dict(), so these checkpoints\n"
      "cannot identify their own arm. The repo name is the provenance record, and\n"
      "re-evaluating a v21 checkpoint needs DECODER=mlp or the strict state_dict load fails.")
