import math
import statistics
import torch
from huggingface_hub import hf_hub_download

# (split_seed, seed) pairs, one per submitted job — an explicit list rather than a
# cross product, so adding the split arm does not conjure rows that were never run.
# To add it: + [(sp, 42) for sp in (43, 44, 45, 46)]
RUNS = [(42, s) for s in (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)]
CACHE = "/Users/kushagrayadav/Code/clinical-graph-jepa/data/fawkes_v20"

# Two evaluation protocols live in every checkpoint and they are NOT comparable.
# recovery_test_loo masks one edge and keeps the whole graph (n=8283) — the paper's
# number. recovery_test_batchmask masks ~30% of edges per batch, caps ranking at
# MRR_CAP=3000, and is the only block carrying AUC/AP.
LOO_METRICS = ("mrr", "hits1", "hits3", "hits10")
BM_METRICS = ("auc", "ap", "auc_nonobvious", "mrr")
# Column headers: auc_nonobvious is wider than its field and would collide with ap.
LABELS = {"mrr": "MRR", "hits1": "H@1", "hits3": "H@3", "hits10": "H@10",
          "auc": "AUC", "ap": "AP", "auc_nonobvious": "AUC_nonobv"}

# baseline/paper_loo_testsplit.json — reproduces the released checkpoint to 0.00e+00.
PUBLISHED_LOO = {"mrr": 0.418653, "hits1": 0.247495, "hits3": 0.473741, "hits10": 0.863576}
# recovery_test_batchmask inside models/fawkes-entity-note/…v16_260615.pt.
PUBLISHED_BM = {"auc": 0.714351, "ap": 0.627698, "auc_nonobvious": 0.800028, "mrr": 0.360833}

loo_rows, bm_rows = {}, {}

for split, seed in RUNS:
    label = f"sp{split}-s{seed}"
    try:
        # per-run local_dir: every repo stores the same fawkes_entity_note.pt filename
        path = hf_hub_download(f"kushagrayadv/fawkes-v20-variance-{label}",
                               "fawkes_entity_note.pt", local_dir=f"{CACHE}/{label}")
    except Exception as exc:
        print(f"{label:<14}  MISSING ({type(exc).__name__}) — check `hf jobs logs`")
        continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    loo_rows[(split, seed)] = ck["recovery_test_loo"]
    if ck.get("recovery_test_batchmask"):
        bm_rows[(split, seed)] = ck["recovery_test_batchmask"]

if not loo_rows:
    raise SystemExit("no runs downloaded — nothing to aggregate")


def mean_sd(values):
    """Sample (n-1) sd. NaN is dropped, not propagated: auc_nonobvious is NaN when a
    run had no non-obvious pairs to score, and one such run must not void the column."""
    clean = [v for v in values if not (v is None or math.isnan(v))]
    if not clean:
        return float("nan"), float("nan"), 0
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return statistics.mean(clean), sd, len(clean)


def summarise(title, rows, metrics, published, extra_key, extra_label):
    print(f"\n{title}")
    print(f"{'run':<14}" + "".join(f"{LABELS.get(m, m):>11}" for m in metrics) + f"{extra_label:>9}")

    def cell(value):
        return f"{'nan':>11}" if value is None or math.isnan(value) else f"{value:>11.4f}"

    for (split, seed), m in rows.items():
        cells = "".join(cell(m.get(k, float("nan"))) for k in metrics)
        print(f"{f'sp{split}-s{seed}':<14}{cells}{m.get(extra_key, ''):>9}")

    agg = {k: mean_sd([m.get(k, float("nan")) for m in rows.values()]) for k in metrics}
    print(f"{'mean':<14}" + "".join(cell(agg[k][0]) for k in metrics))
    print(f"{'sd':<14}" + "".join(cell(agg[k][1]) for k in metrics))
    print(f"{'published':<14}" + "".join(cell(published[k]) for k in metrics))
    print(f"{'Δ mean-pub':<14}" + "".join(f"{agg[k][0] - published[k]:>+11.4f}" for k in metrics))
    for k in metrics:
        if agg[k][2] < len(rows):
            print(f"  note: {k} averaged over {agg[k][2]}/{len(rows)} runs — the rest were NaN")


summarise("LEAVE-ONE-OUT — the paper's protocol, one edge masked, full context",
          loo_rows, LOO_METRICS, PUBLISHED_LOO, "n", "n")

if bm_rows:
    summarise("BATCH-MASK — ~30% of edges masked, ranking capped at MRR_CAP. "
              "The only block with AUC.\nNot comparable with the LOO table above: "
              "different masking, different population.",
              bm_rows, BM_METRICS, PUBLISHED_BM, "n_mrr", "n_mrr")
    # qsig hashes the evaluation questions. The negatives are seeded per graph by gid,
    # not by SEED, so a pinned split should make it identical across all ten runs.
    sigs = sorted({m.get("qsig") for m in bm_rows.values()})
    if len(sigs) == 1:
        print(f"\nquiz_sig identical across all {len(bm_rows)} runs: {sigs[0]} "
              "— the same questions were asked of every model")
    else:
        print(f"\nquiz_sig DIFFERS across runs: {sigs}\n"
              "  The split is pinned, so this should not happen — the arms were not "
              "asked the same questions and the comparison is not paired.")
else:
    print("\nno recovery_test_batchmask block found — AUC/AP unavailable for these runs")

# Which factor the spread comes from. Runs sharing a split are paired — same held-out
# admissions, same denominator. Runs on different splits are not, so that block is a
# spread to report, never a paired test.
observed_splits = sorted({sp for sp, _ in loo_rows})
observed_seeds = sorted({s for _, s in loo_rows})

print("\nseed-to-seed spread, SEED varying at fixed DATA_SPLIT_SEED (same patients, paired):")
for split in observed_splits:
    values = [loo_rows[(split, seed)]["mrr"] for seed in observed_seeds if (split, seed) in loo_rows]
    if len(values) > 1:
        mean, sd, _ = mean_sd(values)
        print(f"  split {split}: LOO MRR {mean:.4f} +/- {sd:.4f}  ({len(values)} seeds)"
              f"  [min {min(values):.4f}, max {max(values):.4f}]")

print("\nsplit-to-split spread, DATA_SPLIT_SEED varying at fixed SEED:")
measured = False
for seed in observed_seeds:
    values = [loo_rows[(sp, seed)]["mrr"] for sp in observed_splits if (sp, seed) in loo_rows]
    if len(values) > 1:
        measured = True
        mean, sd, _ = mean_sd(values)
        print(f"  seed {seed}:  LOO MRR {mean:.4f} +/- {sd:.4f}  ({len(values)} splits)")
if not measured:
    print(f"  NOT MEASURED — every run used DATA_SPLIT_SEED={observed_splits[0]}, so the same\n"
          "  admissions are held out throughout. The std above is initialisation variance\n"
          "  only and does not bound sensitivity to which patients were sampled.")
