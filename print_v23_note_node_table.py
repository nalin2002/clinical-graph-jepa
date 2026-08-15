"""v24 (the global NOTE node) against the two arms it sits between.

v16's claim is that LOCALIZATION, not the note, is what works: the same
Clinical-ModernBERT vector helps when it is written onto the entities the note
grounds and hurts when it hangs off a per-admission NOTE node. That claim rests on
three single runs — no-note 0.274, NOTE node 0.265, entity-grounded 0.419 — and the
step it turns on is the v15 negative, −0.009, which v20 later showed to be half the
seed noise of this pipeline (±0.019). v24 is the missing arm, at ten paired seeds.

Three arms in ONE cell (patch masking + MLP readout, DATA_SPLIT_SEED=42, seeds
42-51), differing only in where the note sits:

    no note              the note is absent
    GROUND_BY=node       one NOTE node per admission carries it   <- v24
    GROUND_BY=prov       the entities the note grounds carry it

which decomposes the note lift exactly:

    (prov − nonote)  =  (node − nonote)  +  (prov − node)
     the note lift       PRESENCE: does      PLACEMENT: what
                         a global note       localization is
                         help at all?        worth

Both halves are what the paper asserts and neither has an error bar. The pairing
controls the split and the seed, not the encoder: the arms differ in node count
(GROUND_BY=node adds one) or in input width (no-note is numeric_dim 6 against 774),
so they are different models — the same caveat v22 and the no-note arms carry.
"""
import math
import statistics

import torch
from huggingface_hub import hf_hub_download

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
CACHE = "data/fawkes_v24"

# t(df, .975), keyed by df = n-1. The sibling tables hardcode 2.262 because they
# assume all ten seeds land; here a single failed job would silently narrow every
# interval, and the verdict this table prints is read straight off the interval.
T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}


def t95(n):
    return T95_BY_DF.get(n - 1, 1.96)

# name -> (placement, owner/repo stem, checkpoint filename)
#
# The node arm went to wmatbooth, the two v22 arms to kushagrayadv, so the owner is
# per-arm rather than one prefix. Its file is fawkes_entity_note.pt like the prov
# arm's: checkpoint_name keys only on use_note, and GLOBAL_NOTE_NODE=1 is still
# USE_NOTE=1. The arm is identified by config["global_note_node"], which the
# provenance block below prints -- do not read the placement off the filename.
ARMS = {
    "nonote": ("none",          "kushagrayadv/fawkes-v22-nonote-patchmlp",   "fawkes_no_note.pt"),
    "node":   ("NOTE node",     "wmatbooth/fawkes-v23-global-note-node",     "fawkes_entity_note.pt"),
    "prov":   ("grounded ents", "kushagrayadv/fawkes-v22-patch-mlp",         "fawkes_entity_note.pt"),
}

# (new arm, baseline arm, what the difference isolates)
COMPARISONS = [
    ("node", "nonote", "PRESENCE -- does a global note node help at all? (v15 said -0.009)"),
    ("prov", "node",   "PLACEMENT -- localization, isolated. The v16 claim."),
    ("prov", "nonote", "the note lift -- presence and placement together"),
]

LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "mrr")          # auc_nonobvious excluded -- see the closing note
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP"}

runs = {}
for arm, (_placement, stem, filename) in ARMS.items():
    for seed in SEEDS:
        label = f"sp{SPLIT}-s{seed}"
        try:
            path = hf_hub_download(f"{stem}-{label}", filename,
                                   local_dir=f"{CACHE}/{arm}-{label}")
        except Exception as exc:
            print(f"{arm:<7} {label}  MISSING ({type(exc).__name__}) -- check `hf jobs ps -a`")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        runs[(arm, seed)] = {"loo": ck["recovery_test_loo"],
                             "bm": ck.get("recovery_test_batchmask"),
                             "config": ck.get("config") or {},
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


def paired_seeds(arms, block):
    return [s for s in SEEDS
            if all((a, s) in runs and runs[(a, s)][block] for a in arms)]


# ---- provenance: the arms must differ in placement and in NOTHING else ----
# use_note, ground_by and global_note_node are in checkpoint_dict, so each file
# states its own placement; mask_strategy and decoder come from run_config. A table
# comparing a patch arm against a random one would otherwise look exactly like this
# one. global_note_node is the load-bearing key: the node arm is USE_NOTE=1 and
# carries the default ground_by=prov, so on the first two fields alone it is
# indistinguishable from the entity-grounded arm.
print("\nPROVENANCE -- what each checkpoint says it is")
cells = set()
for arm in present:
    sample = next(s for s in SEEDS if (arm, s) in runs)
    config, run_config = runs[(arm, sample)]["config"], runs[(arm, sample)]["run_config"]
    placement = (f"use_note={config.get('use_note')} ground_by={config.get('ground_by')}"
                 f" global_note_node={config.get('global_note_node')}")
    if run_config:
        cell = (run_config["mask_strategy"], run_config["decoder"], run_config["data_split_seed"])
        cells.add(cell)
        print(f"  {arm:<7} {placement:<38} mask={cell[0]:<7} decoder={cell[1]:<9} split={cell[2]}")
    else:
        print(f"  {arm:<7} {placement:<38} no run_config -- predates the block, repo name is the record")
if len(cells) > 1:
    print(f"  !! arms are NOT in the same cell: {sorted(cells)} -- the difference below is not placement")

# ---- the three-way ----
print("\nTHREE-WAY -- mean LOO MRR over the seeds present")
print(f"{'placement':<18}{'LOO MRR':>22}")
for arm in ("nonote", "node", "prov"):
    if arm not in ARMS:
        continue
    mean, sd, n = mean_sd(arm_values(arm, "loo", "mrr")) if arm in present else (0, 0, 0)
    measured = f"{mean:.4f}+-{sd:.4f} (n={n})" if n else "MISSING"
    print(f"{ARMS[arm][0]:<18}{measured:>22}")


def report(title, block, metrics):
    print(f"\n{'=' * 86}\n{title}")
    print(f"\n{'arm':<8}{'placement':<17}" + "".join(f"{LABELS[m]:>11}" for m in metrics) + f"{'n':>6}")
    for arm in present:
        vals = "".join(f"{mean_sd(arm_values(arm, block, m))[0]:>11.4f}" for m in metrics)
        sds = "".join(f"{'+-' + format(mean_sd(arm_values(arm, block, m))[1], '.4f'):>11}" for m in metrics)
        n = mean_sd(arm_values(arm, block, metrics[0]))[2]
        print(f"{arm:<8}{ARMS[arm][0]:<17}{vals}{n:>6}")
        print(f"{'':<25}{sds}")

    for new, base, what in COMPARISONS:
        if new not in present or base not in present:
            continue
        seeds = paired_seeds((new, base), block)
        if len(seeds) < 2:
            continue
        print(f"\n  {new} - {base}  ({what})   paired, n={len(seeds)}")
        for m in metrics:
            diffs = [runs[(new, s)][block][m] - runs[(base, s)][block][m] for s in seeds]
            mean, sd, n = mean_sd(diffs)
            half = t95(n) * sd / math.sqrt(n)
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

# ---- the decomposition: how much of the note lift is placement? ----
if all(a in present for a in ("nonote", "node", "prov")):
    seeds = paired_seeds(("nonote", "node", "prov"), "loo")
    if len(seeds) >= 2:
        def delta(new, base):
            return statistics.mean(runs[(new, s)]["loo"]["mrr"] - runs[(base, s)]["loo"]["mrr"]
                                   for s in seeds)
        presence, placement, lift = delta("node", "nonote"), delta("prov", "node"), delta("prov", "nonote")
        print(f"\n{'=' * 86}\nDECOMPOSITION of the note lift -- LOO MRR, {len(seeds)} seeds present in all three arms")
        print(f"  presence   (node - nonote)  {presence:+.4f}")
        print(f"  placement  (prov - node)    {placement:+.4f}")
        print(f"  {'':<27}{'-' * 8}")
        print(f"  note lift  (prov - nonote)  {lift:+.4f}")
        if abs(lift) > 1e-9:
            print(f"\n  placement accounts for {100 * placement / lift:.0f}% of the lift, presence for "
                  f"{100 * presence / lift:.0f}%.")
        print("  v16 asserts placement is effectively all of it and presence is NEGATIVE. Read the two")
        print("  confidence intervals above before repeating that: a presence term whose CI spans zero")
        print("  means the NOTE node was never shown to hurt, only shown not to help.")

print(f"\n{'=' * 86}")
print("auc_nonobvious is NOT reported. The negative-pool defect was fixed on 260809, so the")
print("v24 arm carries the corrected metric and the v22 note arm (pushed earlier) does not --")
print("this table would compare two different metrics. Use print_nonobvious_rescore_table.py")
print("to re-score before quoting a non-obvious number across these three arms.")
print("\nv20 put this pipeline's seed noise at +-0.019 LOO MRR. Any arm difference smaller than")
print("that is a coin flip at one seed, which is what the v15 negative (-0.009) was.")
