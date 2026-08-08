"""v22 (patch mask + MLP) against the arms it is paired with.

Every arm shares DATA_SPLIT_SEED=42 and seeds 42-51, so all comparisons are paired:
same held-out admissions, same 8,283 LOO edges, same denominator. Add an arm to ARMS
and it is picked up everywhere, including the design grid.
"""
import math
import statistics
import torch
from huggingface_hub import hf_hub_download

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
CACHE = "/Users/kushagrayadav/Code/clinical-graph-jepa/data/fawkes_v22"

# name -> (mask strategy, decoder, repo stem)
ARMS = {
    "v20": ("random", "distmult", "fawkes-v20-variance"),
    "v21": ("random", "mlp",      "fawkes-v21-mlp"),
    "v22": ("patch",  "mlp",      "fawkes-v22-patch-mlp"),
    # The fourth cell. Uncomment once submitted; everything below adapts.
    # "v23": ("patch", "distmult", "fawkes-v23-patch-distmult"),
}

# (new arm, baseline arm, what the difference isolates)
COMPARISONS = [
    ("v22", "v21", "mask strategy, at the MLP head"),
    ("v22", "v20", "the full stack, against the published configuration"),
    ("v21", "v20", "readout head, at random masking (v21, for reference)"),
]

LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "mrr")          # auc_nonobvious deliberately excluded — see note
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP"}
T95 = 2.262   # t(9, .975)

runs = {}
for arm, (_mask, _dec, stem) in ARMS.items():
    for seed in SEEDS:
        label = f"sp{SPLIT}-s{seed}"
        try:
            path = hf_hub_download(f"kushagrayadv/{stem}-{label}", "fawkes_entity_note.pt",
                                   local_dir=f"{CACHE}/{arm}-{label}")
        except Exception as exc:
            print(f"{arm:<5} {label}  MISSING ({type(exc).__name__}) — check `hf jobs logs`")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        runs[(arm, seed)] = {"loo": ck["recovery_test_loo"],
                             "bm": ck.get("recovery_test_batchmask")}

present = [a for a in ARMS if any((a, s) in runs for s in SEEDS)]
if not present:
    raise SystemExit("no checkpoints downloaded — nothing to report")


def mean_sd(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def arm_values(arm, block, metric):
    return [runs[(arm, s)][block][metric] for s in SEEDS
            if (arm, s) in runs and runs[(arm, s)][block]]


# ---- the design, so a missing cell is visible rather than assumed ----
print("\nDESIGN — mean LOO MRR per cell")
decoders = sorted({d for _m, d, _r in ARMS.values()})
masks = sorted({m for m, _d, _r in ARMS.values()})
print(f"{'':<10}" + "".join(f"{d:>18}" for d in decoders))
for mask in masks:
    cells = []
    for dec in decoders:
        arm = next((a for a, (m, d, _r) in ARMS.items() if m == mask and d == dec), None)
        if arm is None:
            cells.append(f"{'(not run)':>18}")
        else:
            mean, sd, n = mean_sd(arm_values(arm, "loo", "mrr"))
            cells.append(f"{f'{arm} {mean:.4f}±{sd:.4f}':>18}" if n else f"{f'{arm} MISSING':>18}")
    print(f"{mask:<10}" + "".join(cells))


def report(title, block, metrics):
    print(f"\n{'=' * 78}\n{title}")
    print(f"\n{'arm':<6}{'mask':<9}{'decoder':<10}" + "".join(f"{LABELS[m]:>11}" for m in metrics) + f"{'n':>7}")
    for arm in present:
        mask, dec, _ = ARMS[arm]
        vals = "".join(f"{mean_sd(arm_values(arm, block, m))[0]:>11.4f}" for m in metrics)
        sds = "".join(f"{'±' + format(mean_sd(arm_values(arm, block, m))[1], '.4f'):>11}" for m in metrics)
        n = mean_sd(arm_values(arm, block, metrics[0]))[2]
        print(f"{arm:<6}{mask:<9}{dec:<10}{vals}{n:>7}")
        print(f"{'':<25}{sds}")

    for new, base, what in COMPARISONS:
        if new not in present or base not in present:
            continue
        seeds = [s for s in SEEDS if (new, s) in runs and (base, s) in runs
                 and runs[(new, s)][block] and runs[(base, s)][block]]
        if len(seeds) < 2:
            continue
        print(f"\n  {new} − {base}  ({what})   paired, n={len(seeds)}")
        for m in metrics:
            diffs = [runs[(new, s)][block][m] - runs[(base, s)][block][m] for s in seeds]
            mean, sd, n = mean_sd(diffs)
            half = T95 * sd / math.sqrt(n)
            lo, hi = mean - half, mean + half
            # A paired 95% CI excluding zero is the two-sided paired t-test at .05.
            verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
            wins = sum(1 for d in diffs if d > 0)
            print(f"    {LABELS[m]:<6} {mean:+.4f} ± {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {verdict:<26} ahead at {wins}/{n}")


report("LEAVE-ONE-OUT — the paper's protocol, one edge masked, full context", "loo", LOO_METRICS)
if all(runs[(a, s)]["bm"] for a in present for s in SEEDS if (a, s) in runs):
    report("BATCH-MASK — ~30% of edges masked, capped at MRR_CAP. "
           "Not comparable with the LOO block above.", "bm", BM_METRICS)
else:
    print("\nbatch-mask block missing from at least one run — skipped")

print(f"\n{'=' * 78}")
print("auc_nonobvious is NOT reported. _classification_metrics filters the positives to")
print("non-patient-anchored edges but scores them against the unfiltered negative pool,")
print("which is ~68% patient-anchored. The v21 report reproduced the resulting sign error.")
print("Fix it (negatives[nonpat_mask]) and re-score before quoting that column anywhere.")

missing = [(m, d) for m in masks for d in decoders
           if not any(mm == m and dd == d for mm, dd, _ in ARMS.values())]
if missing:
    print(f"\nDesign is incomplete: {', '.join(f'({m}, {d})' for m, d in missing)} not run.")
    print("A mask-strategy effect measured only at one decoder cannot be separated from an")
    print("interaction between the two. Fill the cell before claiming patch masking helps")
    print("or hurts in general rather than specifically with this head.")

print("\nNone of MASK_STRATEGY, JEPA_PATCHES, DECODER or DATA_SPLIT_SEED is written into")
print("checkpoint_dict(), so these files cannot identify their own arm — the repo name is")
print("the provenance record. Re-evaluating any mlp arm needs DECODER=mlp set.")
