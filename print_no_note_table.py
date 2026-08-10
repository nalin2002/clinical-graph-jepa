"""The no-note arms of v20/v21/v22, against each other and against their note twins.

Two questions this answers that the note-only 2x2 could not:

1. Does v21's readout-head win survive without the note? The +0.0530 this project
   adopted was only ever measured with the 768-d Clinical-ModernBERT channel
   present, so "the MLP head is better" and "the MLP head is better at exploiting
   the note" are not yet separable.
2. Does v18's ONLY positive result hold at power? v18 reported no-note patch
   masking at +0.0237 MRR on one seed, and built its mechanism argument on it —
   that patch masking teaches the structure-only model what the note was otherwise
   supplying. v20 later put the noise floor at +-0.019. v22 gave the with-note arm
   ten paired seeds and found nothing; the no-note arm never got them.

Every arm shares DATA_SPLIT_SEED=42 and seeds 42-51, so all comparisons are paired
on the same held-out admissions. The note and no-note arms differ in the encoder's
input width (numeric_dim 774 vs 6), so those pairs control the split and the seed,
not the encoder — same caveat as v22's mask-strategy comparison.
"""
import math
import statistics

import torch
from huggingface_hub import hf_hub_download

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
CACHE = "data/fawkes_no_note"
T95 = 2.262   # t(9, .975)

# name -> (note?, mask, decoder, repo stem, checkpoint filename)
ARMS = {
    "v20":        (True,  "random", "distmult", "fawkes-v20-variance",          "fawkes_entity_note.pt"),
    "v21":        (True,  "random", "mlp",      "fawkes-v21-mlp",               "fawkes_entity_note.pt"),
    "v22":        (True,  "patch",  "mlp",      "fawkes-v22-patch-mlp",         "fawkes_entity_note.pt"),
    "v20-nonote": (False, "random", "distmult", "fawkes-v20-nonote",            "fawkes_no_note.pt"),
    "v21-nonote": (False, "random", "mlp",      "fawkes-v21-nonote-mlp",        "fawkes_no_note.pt"),
    "v22-nonote": (False, "patch",  "mlp",      "fawkes-v22-nonote-patchmlp",   "fawkes_no_note.pt"),
}

# (new arm, baseline arm, what the difference isolates)
COMPARISONS = [
    ("v21-nonote", "v20-nonote", "readout head, WITHOUT the note -- does v21 generalise?"),
    ("v22-nonote", "v21-nonote", "mask strategy, WITHOUT the note -- v18's positive result, at power"),
    ("v22-nonote", "v20-nonote", "the full stack, without the note"),
    ("v20", "v20-nonote", "the NOTE LIFT at the published configuration"),
    ("v21", "v21-nonote", "the NOTE LIFT at the MLP head"),
    ("v22", "v22-nonote", "the NOTE LIFT under patch masking"),
]

LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "auc_nonobvious", "mrr")
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP", "auc_nonobvious": "nonobv"}

runs = {}
for arm, (_note, _mask, _dec, stem, filename) in ARMS.items():
    for seed in SEEDS:
        label = f"sp{SPLIT}-s{seed}"
        try:
            path = hf_hub_download(f"kushagrayadv/{stem}-{label}", filename,
                                   local_dir=f"{CACHE}/{arm}-{label}")
        except Exception as exc:
            print(f"{arm:<11} {label}  MISSING ({type(exc).__name__}) -- check `hf jobs ps -a`")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        runs[(arm, seed)] = {"loo": ck["recovery_test_loo"],
                             "bm": ck.get("recovery_test_batchmask"),
                             # (260809) the no-note arms are the first to carry this.
                             "run_config": ck.get("run_config")}

present = [a for a in ARMS if any((a, s) in runs for s in SEEDS)]
if not present:
    raise SystemExit("no checkpoints downloaded -- nothing to report")


def mean_sd(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def arm_values(arm, block, metric):
    return [runs[(arm, s)][block][metric] for s in SEEDS
            if (arm, s) in runs and runs[(arm, s)][block]]


# ---- provenance: the point of the run_config block ----
print("\nPROVENANCE -- does each checkpoint identify its own arm?")
for arm in present:
    sample = next((runs[(arm, s)]["run_config"] for s in SEEDS if (arm, s) in runs), None)
    if sample is None:
        print(f"  {arm:<11} no run_config -- predates the block; the repo name is the record")
    else:
        print(f"  {arm:<11} mask={sample['mask_strategy']:<7} decoder={sample['decoder']:<9} "
              f"split={sample['data_split_seed']} frozen={sample['freeze_encoder']}")

# ---- the design, so a missing cell is visible rather than assumed ----
print("\nDESIGN -- mean LOO MRR per cell")
print(f"{'':<16}{'random/distmult':>20}{'random/mlp':>20}{'patch/mlp':>20}")
for note_flag, row_label in ((True, "with note"), (False, "no note")):
    cells = []
    for mask, dec in (("random", "distmult"), ("random", "mlp"), ("patch", "mlp")):
        arm = next((a for a, (n, m, d, _s, _f) in ARMS.items()
                    if n == note_flag and m == mask and d == dec), None)
        mean, sd, n = mean_sd(arm_values(arm, "loo", "mrr")) if arm else (0, 0, 0)
        cells.append(f"{f'{mean:.4f}+-{sd:.4f}' if n else 'MISSING':>20}")
    print(f"{row_label:<16}" + "".join(cells))


def report(title, block, metrics):
    print(f"\n{'=' * 86}\n{title}")
    print(f"\n{'arm':<12}{'note':<6}{'mask':<8}{'decoder':<10}"
          + "".join(f"{LABELS[m]:>11}" for m in metrics) + f"{'n':>6}")
    for arm in present:
        note_flag, mask, dec, _s, _f = ARMS[arm]
        vals = "".join(f"{mean_sd(arm_values(arm, block, m))[0]:>11.4f}" for m in metrics)
        sds = "".join(f"{'+-' + format(mean_sd(arm_values(arm, block, m))[1], '.4f'):>11}" for m in metrics)
        n = mean_sd(arm_values(arm, block, metrics[0]))[2]
        print(f"{arm:<12}{'yes' if note_flag else 'no':<6}{mask:<8}{dec:<10}{vals}{n:>6}")
        print(f"{'':<36}{sds}")

    for new, base, what in COMPARISONS:
        if new not in present or base not in present:
            continue
        seeds = [s for s in SEEDS if (new, s) in runs and (base, s) in runs
                 and runs[(new, s)][block] and runs[(base, s)][block]]
        if len(seeds) < 2:
            continue
        print(f"\n  {new} - {base}  ({what})   paired, n={len(seeds)}")
        for m in metrics:
            diffs = [runs[(new, s)][block][m] - runs[(base, s)][block][m] for s in seeds]
            mean, sd, n = mean_sd(diffs)
            half = T95 * sd / math.sqrt(n)
            lo, hi = mean - half, mean + half
            # A paired 95% CI excluding zero is the two-sided paired t-test at .05.
            verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
            wins = sum(1 for d in diffs if d > 0)
            print(f"    {LABELS[m]:<7} {mean:+.4f} +- {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {verdict:<26} ahead at {wins}/{n}")


report("LEAVE-ONE-OUT -- the paper's protocol, one edge masked, full context", "loo", LOO_METRICS)
if all(runs[(a, s)]["bm"] for a in present for s in SEEDS if (a, s) in runs):
    report("BATCH-MASK -- ~30% of edges masked, capped at MRR_CAP. "
           "Not comparable with the LOO block above.", "bm", BM_METRICS)
else:
    print("\nbatch-mask block missing from at least one run -- skipped")

print(f"\n{'=' * 86}")
print("auc_nonobvious IS reported here: the negative-pool defect was fixed on 260809")
print("(negatives[nonpat_mask]) and the note arms re-scored. Any value read from a")
print("checkpoint pushed BEFORE that date is still the inflated one -- the no-note arms")
print("carry the corrected metric, the note arms do not. Re-score with")
print("print_nonobvious_rescore_table.py before quoting the two side by side.")
print("\nv18 measured the no-note deltas at one seed: patch-vs-random MRR +0.0237,")
print("H@1 +0.0221, H@3 +0.0349, H@10 +0.0136, and the note lift at +0.144 narrowing")
print("to +0.110. Those are the numbers the v22-nonote - v21-nonote row above tests.")
