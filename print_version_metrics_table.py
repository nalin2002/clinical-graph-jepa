"""The complete metric set for v20/v21/v22, both note variants, as the report prints it.

Seven metrics per arm -- LOO MRR/H@1/H@3/H@10 and batch-mask AUC/non-obvious AUC/MRR --
plus per-relation LOO MRR, so no version is documented on a different subset of
measurements than its neighbours.

One column needs re-computing rather than reading. `auc_nonobvious` was defective
until 260809: it filtered the positives to non-PATIENT-anchored edges but scored them
against the whole negative pool. The note arms were trained before the fix and their
stored value is the inflated one; the no-note arms trained after it. Reading the two
side by side would compare two different metrics, so this script re-scores every arm
on CPU and reports the corrected column -- device-consistent, which is what makes the
note-vs-no-note contrast legitimate rather than merely both-corrected.

Superseded print_no_note_table.py, which covered the same arms on a narrower metric set.
"""
import math
import statistics
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from fawkes.config import Config
from fawkes.model import Encoder, build_scorer
from fawkes.steps import readout_step
from fawkes.train import prepare_data

DATA = "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SPLIT = 42
T95 = 2.262   # t(9, .975)

# version -> use_note -> (mask, decoder, local cache stem, checkpoint filename)
ARMS = {
    "v20": {True:  ("random", "distmult", "data/fawkes_v20/",                  "fawkes_entity_note.pt"),
            False: ("random", "distmult", "data/fawkes_no_note/v20-nonote-",   "fawkes_no_note.pt")},
    "v21": {True:  ("random", "mlp",      "data/fawkes_v21/mlp-",              "fawkes_entity_note.pt"),
            False: ("random", "mlp",      "data/fawkes_no_note/v21-nonote-",   "fawkes_no_note.pt")},
    "v22": {True:  ("patch",  "mlp",      "data/fawkes_v22/v22-",              "fawkes_entity_note.pt"),
            False: ("patch",  "mlp",      "data/fawkes_no_note/v22-nonote-",   "fawkes_no_note.pt")},
}


@torch.no_grad()
def corrected_nonobvious(encoder, scorer, loader, device, cfg):
    """auc_nonobvious with the negative pool masked to match the positives."""
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
    p, n = positives[mask], negatives[mask]
    return roc_auc_score(np.concatenate([np.ones_like(p), np.zeros_like(n)]), np.concatenate([p, n]))


def mean_sd(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def paired(new, base, key):
    seeds = [s for s in SEEDS if (new, s) in runs and (base, s) in runs]
    diffs = [runs[(new, s)][key] - runs[(base, s)][key] for s in seeds]
    mean, sd, n = mean_sd(diffs)
    half = T95 * sd / math.sqrt(n) if n > 1 else float("nan")
    wins = sum(1 for d in diffs if d > 0)
    return mean, sd, mean - half, mean + half, wins, n


device = torch.device("cpu")
# to_data folds the note into the node features, so each variant needs its own pass.
LOADERS = {}
for use_note in (True, False):
    _tr, _va, test_pairs = prepare_data(Config(data_path=DATA, data_split_seed=SPLIT, use_note=use_note))
    LOADERS[use_note] = DataLoader([d for d, _ in test_pairs], batch_size=1, shuffle=False)
    print(f"TEST graphs={len(LOADERS[use_note].dataset)} use_note={use_note} split={SPLIT} device=cpu")

runs, per_rel = {}, defaultdict(lambda: defaultdict(list))
for version, variants in ARMS.items():
    for use_note, (mask, decoder, stem, filename) in variants.items():
        arm = f"{version}{'' if use_note else '-nonote'}"
        cfg = Config(data_path=DATA, data_split_seed=SPLIT, decoder=decoder, use_note=use_note)
        for seed in SEEDS:
            path = f"{stem}sp{SPLIT}-s{seed}/{filename}"
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
            except FileNotFoundError:
                print(f"{arm:<12} s{seed}  MISSING {path}")
                continue
            encoder, scorer = Encoder(cfg).to(device), build_scorer(cfg).to(device)
            encoder.load_state_dict(ck["encoder"])
            scorer.load_state_dict(ck["scorer"])
            loo, bm = ck["recovery_test_loo"], ck["recovery_test_batchmask"]
            runs[(arm, seed)] = {
                "loo_mrr": loo["mrr"], "loo_h1": loo["hits1"],
                "loo_h3": loo["hits3"], "loo_h10": loo["hits10"],
                "bm_auc": bm["auc"], "bm_mrr": bm["mrr"],
                "nonobv_stored": bm["auc_nonobvious"],
                "nonobv": corrected_nonobvious(encoder, scorer, LOADERS[use_note], device, cfg),
            }
            for row in loo["per_rel"]:
                per_rel[arm][row["rel"]].append((row["mrr"], row["n"], row["chance_mrr"]))
        print(f"{arm:<12} collected {sum(1 for s in SEEDS if (arm, s) in runs)}/10")

COLS = [("loo_mrr", "LOO MRR"), ("loo_h1", "LOO H@1"), ("loo_h3", "LOO H@3"), ("loo_h10", "LOO H@10"),
        ("bm_auc", "BM AUC"), ("nonobv", "nonobv*"), ("bm_mrr", "BM MRR")]

for version in ARMS:
    print(f"\n{'=' * 100}\n{version} -- all seven metrics, mean +- sd over 10 seeds"
          f"   (* nonobv re-scored on CPU with the corrected metric)")
    print(f"\n{'variant':<10}" + "".join(f"{label:>13}" for _k, label in COLS))
    for arm, label in ((version, "with note"), (f"{version}-nonote", "no note")):
        if not any((arm, s) in runs for s in SEEDS):
            continue
        vals = "".join(f"{mean_sd([runs[(arm, s)][k] for s in SEEDS if (arm, s) in runs])[0]:>13.4f}"
                       for k, _l in COLS)
        sds = "".join(f"{'+-' + format(mean_sd([runs[(arm, s)][k] for s in SEEDS if (arm, s) in runs])[1], '.4f'):>13}"
                      for k, _l in COLS)
        print(f"{label:<10}{vals}")
        print(f"{'':<10}{sds}")
    note_arm, no_note_arm = version, f"{version}-nonote"
    if any((no_note_arm, s) in runs for s in SEEDS):
        print(f"\n  NOTE LIFT (with - without), paired")
        for k, label in COLS:
            mean, sd, lo, hi, wins, n = paired(note_arm, no_note_arm, k)
            verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
            print(f"    {label:<9} {mean:+.4f} +- {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{verdict:<26} ahead at {wins}/{n}")
    stored = mean_sd([runs[(version, s)]["nonobv_stored"] for s in SEEDS if (version, s) in runs])[0]
    fixed = mean_sd([runs[(version, s)]["nonobv"] for s in SEEDS if (version, s) in runs])[0]
    print(f"\n  nonobv as stored in the with-note checkpoints: {stored:.4f}  ->  corrected {fixed:.4f} "
          f"({fixed - stored:+.4f})")

print(f"\n{'=' * 100}\nVERSION-TO-VERSION, WITHIN EACH NOTE SETTING -- the switch each version turns,")
print("measured separately with and without the note. This is what the report's main tables show.")
for new, base, what in (("v21", "v20", "readout head: DistMult -> MLP"),
                        ("v22", "v21", "pretext task: random -> patch")):
    for suffix, label in (("", "WITH note"), ("-nonote", "NO note")):
        a, b = f"{new}{suffix}", f"{base}{suffix}"
        if not any((a, s) in runs for s in SEEDS) or not any((b, s) in runs for s in SEEDS):
            continue
        print(f"\n  {a} - {b}   ({what}, {label})")
        for k, label_m in COLS:
            mean, sd, lo, hi, wins, n = paired(a, b, k)
            verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not distinguishable from 0"
            print(f"    {label_m:<9} {mean:+.4f} +- {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{verdict:<26} ahead at {wins}/{n}")

print(f"\n{'=' * 100}\nPER-RELATION LOO MRR -- mean over 10 seeds. These relations are the whole LOO population.")
relations = ["MANAGED_FOR", "INDICATES", "COMPLICATED_BY", "CONFIRMS", "TARGETS_ORGANISM"]
arms = [a for v in ARMS for a in (v, f"{v}-nonote") if a in per_rel]
print(f"\n{'relation':<18}{'n':>7}{'chance':>9}" + "".join(f"{a:>20}" for a in arms))
for rel in relations:
    cells, n_q, chance = [], None, None
    for arm in arms:
        rows = per_rel[arm].get(rel, [])
        if rows:
            n_q, chance = rows[0][1], rows[0][2]
            mean, sd, _n = mean_sd([r[0] for r in rows])
            cells.append(f"{f'{mean:.4f}+-{sd:.4f}':>20}")
        else:
            cells.append(f"{'--':>20}")
    print(f"{rel:<18}{n_q if n_q else '--':>7}{chance if chance else float('nan'):>9.3f}" + "".join(cells))

print("\nPer-relation NOTE LIFT (with - without), paired per seed:")
for version in ARMS:
    if f"{version}-nonote" not in per_rel:
        continue
    print(f"\n  {version}: with note - no note")
    for rel in relations:
        a, b = per_rel[version].get(rel), per_rel[f"{version}-nonote"].get(rel)
        if not a or not b:
            continue
        diffs = [x[0] - y[0] for x, y in zip(a, b)]
        mean, sd, n = mean_sd(diffs)
        wins = sum(1 for d in diffs if d > 0)
        print(f"    {rel:<18} {mean:+.4f} +- {sd:.4f}  ahead at {wins}/{n}   (n_queries={a[0][1]})")

print("\nPer-relation paired deltas that the report quotes:")
for new, base, what in (("v21", "v20", "readout head, with note"),
                        ("v21-nonote", "v20-nonote", "readout head, no note"),
                        ("v22", "v21", "mask strategy, with note"),
                        ("v22-nonote", "v21-nonote", "mask strategy, no note")):
    if new not in per_rel or base not in per_rel:
        continue
    print(f"\n  {new} - {base}  ({what})")
    for rel in relations:
        a, b = per_rel[new].get(rel), per_rel[base].get(rel)
        if not a or not b:
            continue
        diffs = [x[0] - y[0] for x, y in zip(a, b)]
        mean, sd, n = mean_sd(diffs)
        wins = sum(1 for d in diffs if d > 0)
        print(f"    {rel:<18} {mean:+.4f} +- {sd:.4f}  ahead at {wins}/{n}   (n_queries={a[0][1]})")
