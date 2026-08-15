"""v23 (note injection) against the arms it is paired with.

Every arm shares DATA_SPLIT_SEED=42 and seeds 42-51, so all comparisons are paired:
same held-out admissions, same LOO edges, same denominator. Add an arm to ARMS and it
is picked up everywhere, including the design list and the per-relation block.

The three arms differ only in how the note is POOLED before it reaches the grounded
entities -- placement, grounding, masking, readout and schedule are v22's throughout.
`mean` is the v22 path retrained on the same hardware as the other two.

Reading nalin9's published runs instead of your own is a one-word change: OWNER.
"""
import math
import statistics
import torch
from huggingface_hub import hf_hub_download

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
CACHE = "data/fawkes_v23"
OWNER = "wmatbooth"          # nalin9 for the published v23 aggregate

# name -> (pooling, repo stem)
ARMS = {
    "mean":      ("global mean",   "fawkes-v23-mean-note"),
    "uniform":   ("uniform spans", "fawkes-v23-uniform-note"),
    "attention": ("entity attn",   "fawkes-v23-attention-note"),
}

# (new arm, baseline arm, what the difference isolates)
COMPARISONS = [
    ("uniform", "mean", "span pooling, against the stored global mean"),
    ("attention", "mean", "the v23 arm -- entity-conditioned attention over spans"),
    ("attention", "uniform", "attention, against its own pooling control"),
]

LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "mrr")          # auc_nonobvious deliberately excluded -- see note
TARGET_RELS = ("MANAGED_FOR", "INDICATES", "COMPLICATED_BY", "CONFIRMS")
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP"}
T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}

runs = {}
for arm, (_pooling, stem) in ARMS.items():
    for seed in SEEDS:
        label = f"sp{SPLIT}-s{seed}"
        try:
            path = hf_hub_download(f"{OWNER}/{stem}-{label}", "fawkes_entity_note.pt",
                                   local_dir=f"{CACHE}/{arm}-{label}")
        except Exception as exc:
            print(f"{arm:<10} {label}  MISSING ({type(exc).__name__}) -- check `hf jobs ps -a`")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        runs[(arm, seed)] = {"loo": ck["recovery_test_loo"],
                             "bm": ck.get("recovery_test_batchmask"),
                             "config": ck.get("config") or {}}

present = [a for a in ARMS if any((a, s) in runs for s in SEEDS)]
if not present:
    raise SystemExit(f"no checkpoints downloaded for owner {OWNER!r} -- nothing to report")


def t95(n):
    return T95_BY_DF.get(n - 1, 1.96)


def mean_sd(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def arm_values(arm, block, metric):
    return [runs[(arm, s)][block][metric] for s in SEEDS
            if (arm, s) in runs and runs[(arm, s)][block]]


def rel_values(arm, relation):
    out = []
    for s in SEEDS:
        if (arm, s) not in runs:
            continue
        rows = {row["rel"]: row for row in runs[(arm, s)]["loo"]["per_rel"]}
        if relation in rows:
            out.append(rows[relation]["mrr"])
    return out


def paired_diffs(new, base, values_of):
    """Per-seed differences over the seeds where BOTH arms landed."""
    return [n - b for s in SEEDS
            for n, b in [(values_of(new, s), values_of(base, s))]
            if n is not None and b is not None]


def verdict_line(label, diffs, width=7):
    mean, sd, n = mean_sd(diffs)
    half = t95(n) * sd / math.sqrt(n) if n > 1 else float("nan")
    lo, hi = mean - half, mean + half
    # A paired 95% CI excluding zero is the two-sided paired t-test at .05.
    verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
    wins = sum(1 for d in diffs if d > 0)
    print(f"    {label:<{width}} {mean:+.4f} +- {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  {verdict:<26} ahead at {wins}/{n}")


# ---- the design, so a missing arm is visible rather than assumed ----
print("\nDESIGN -- mean LOO MRR per pooling")
for arm in ARMS:
    mean, sd, n = mean_sd(arm_values(arm, "loo", "mrr")) if arm in present else (0, 0, 0)
    cell = f"{mean:.4f}+-{sd:.4f}" if n else "MISSING"
    print(f"  {arm:<11}{ARMS[arm][0]:<16}{cell:>20}{f'n={n}':>7}")


def report(title, block, metrics):
    print(f"\n{'=' * 86}\n{title}")
    print(f"\n{'arm':<11}{'pooling':<16}" + "".join(f"{LABELS[m]:>11}" for m in metrics) + f"{'n':>6}")
    for arm in present:
        vals = "".join(f"{mean_sd(arm_values(arm, block, m))[0]:>11.4f}" for m in metrics)
        sds = "".join(f"{'+-' + format(mean_sd(arm_values(arm, block, m))[1], '.4f'):>11}" for m in metrics)
        n = mean_sd(arm_values(arm, block, metrics[0]))[2]
        print(f"{arm:<11}{ARMS[arm][0]:<16}{vals}{n:>6}")
        print(f"{'':<27}{sds}")

    for new, base, what in COMPARISONS:
        if new not in present or base not in present:
            continue
        seeds = [s for s in SEEDS if (new, s) in runs and (base, s) in runs
                 and runs[(new, s)][block] and runs[(base, s)][block]]
        if len(seeds) < 2:
            continue
        print(f"\n  {new} - {base}  ({what})   paired, n={len(seeds)}")
        for m in metrics:
            verdict_line(LABELS[m],
                         [runs[(new, s)][block][m] - runs[(base, s)][block][m] for s in seeds])


report("LEAVE-ONE-OUT -- the paper's protocol, one edge masked, full context", "loo", LOO_METRICS)
if all(runs[(a, s)]["bm"] for a in present for s in SEEDS if (a, s) in runs):
    report("BATCH-MASK -- ~30% of edges masked, capped at MRR_CAP. "
           "Not comparable with the LOO block above.", "bm", BM_METRICS)
else:
    print("\nbatch-mask block missing from at least one run -- skipped")

# ---- per inferred relation: where a pooling change should show up if anywhere ----
print(f"\n{'=' * 86}\nLOO PER INFERRED RELATION -- MRR. The note narrates these four; a")
print("pooling change that does nothing here did nothing anywhere.")
print(f"\n{'arm':<11}" + "".join(f"{r:>18}" for r in TARGET_RELS))
for arm in present:
    cells = []
    for relation in TARGET_RELS:
        mean, sd, n = mean_sd(rel_values(arm, relation))
        cells.append(f"{f'{mean:.4f}+-{sd:.4f}' if n else 'MISSING':>18}")
    print(f"{arm:<11}" + "".join(cells))

for new, base, _what in COMPARISONS:
    if new not in present or base not in present:
        continue
    print(f"\n  {new} - {base}   per relation, paired")
    for relation in TARGET_RELS:
        def mrr_of(arm, seed, relation=relation):
            if (arm, seed) not in runs:
                return None
            rows = {row["rel"]: row for row in runs[(arm, seed)]["loo"]["per_rel"]}
            return rows[relation]["mrr"] if relation in rows else None
        diffs = paired_diffs(new, base, mrr_of)
        if len(diffs) > 1:
            verdict_line(relation, diffs, width=16)

print(f"\n{'=' * 86}")
print("auc_nonobvious is NOT reported. _classification_metrics filters the positives to")
print("non-patient-anchored edges but scores them against the unfiltered negative pool.")
print("Re-score with print_nonobvious_rescore_table.py before quoting that column.")
print("\nAll three arms are USE_NOTE=1 GROUND_BY=prov. There is no v23 no-note arm --")
print("the no-note results belong to v20/v21/v22 and print_no_note_table.py reads them.")
print(f"\nOWNER={OWNER}. submit-v23-note-attention.sh pushes to $(hf auth whoami), so a rerun")
print("under another account writes repos this script will not find.")
