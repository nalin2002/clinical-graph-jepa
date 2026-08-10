"""Floor / cascade / ceiling per inferred relation, for arms that never ran RUN_CASCADE=1.

Every run from v19 onward was submitted with RUN_CASCADE=0 to save GPU time, so the
`cascade` block in all thirty v20/v21/v22 checkpoints is None. The evaluator itself is
untouched, and the weights are on disk, so the numbers can be recovered locally.

The three quantities, all LOO MRR per relation:

  floor    context = the deterministic backbone only. Every inferred cross-link is
           removed, so this is what the relation is worth without the other inferred
           edges to lean on.
  cascade  context = backbone plus the GOLD edges of every relation earlier in
           CASCADE_ORDER (an oracle cascade), so each relation sees the ones before it.
  ceiling  context = the whole graph bar the single masked edge -- i.e. the ordinary
           leave-one-out number the report already quotes.

ceiling - floor is the headroom that surrounding context is worth. v18 found it
NEGATIVE for two of four relations in the no-note model, meaning context actively
confused the encoder; that is the measurement this script generalises to the
ten-seed arms.

SEEDS defaults to a single seed because the cascade is ~8 context rebuilds per graph;
widen it once you have a timing.
"""
import argparse
import statistics
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

from fawkes.config import Config
from fawkes.data import RELATION_CANONICAL, TARGET_RELS
from fawkes.evaluate import cascade_evaluate, loo_evaluate
from fawkes.model import Encoder, build_scorer
from fawkes.train import prepare_data

DATA = "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"
SPLIT = 42

ARMS = {
    "v22":        (True,  "patch",  "mlp",      "data/fawkes_v22/v22-",            "fawkes_entity_note.pt"),
    "v22-nonote": (False, "patch",  "mlp",      "data/fawkes_no_note/v22-nonote-", "fawkes_no_note.pt"),
    "v21":        (True,  "random", "mlp",      "data/fawkes_v21/mlp-",            "fawkes_entity_note.pt"),
    "v21-nonote": (False, "random", "mlp",      "data/fawkes_no_note/v21-nonote-", "fawkes_no_note.pt"),
    "v20":        (True,  "random", "distmult", "data/fawkes_v20/",                "fawkes_entity_note.pt"),
    "v20-nonote": (False, "random", "distmult", "data/fawkes_no_note/v20-nonote-", "fawkes_no_note.pt"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--arms", default="v22", help="comma-separated arm names")
parser.add_argument("--seeds", default="42", help="comma-separated training seeds")
args = parser.parse_args()
arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

device = torch.device("cpu")
GRAPHS = {}
for use_note in {ARMS[a][0] for a in arm_names}:
    _tr, _va, test_pairs = prepare_data(Config(data_path=DATA, data_split_seed=SPLIT, use_note=use_note))
    GRAPHS[use_note] = [d for d, _ in test_pairs]
    print(f"TEST graphs={len(GRAPHS[use_note])} use_note={use_note} split={SPLIT} device=cpu")

results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # arm -> rel -> stage -> [values]
for arm in arm_names:
    use_note, mask, decoder, stem, filename = ARMS[arm]
    cfg = Config(data_path=DATA, data_split_seed=SPLIT, decoder=decoder, use_note=use_note)
    order_ids = [RELATION_CANONICAL[r] for r in cfg.cascade_order]
    for seed in seeds:
        ck = torch.load(f"{stem}sp{SPLIT}-s{seed}/{filename}", map_location="cpu", weights_only=False)
        encoder, scorer = Encoder(cfg).to(device), build_scorer(cfg).to(device)
        encoder.load_state_dict(ck["encoder"])
        scorer.load_state_dict(ck["scorer"])

        casc = cascade_evaluate(encoder, scorer, GRAPHS[use_note], order_ids, device, cfg)
        # ceiling: the LOO number already stored in the checkpoint, per relation
        ceiling = {row["rel"]: row["mrr"] for row in ck["recovery_test_loo"]["per_rel"]}
        for rel in TARGET_RELS:
            for stage, block in (("floor", casc.get("floor", {})), ("cascade", casc.get("cascade", {}))):
                row = block.get(rel)
                if isinstance(row, dict) and "mrr" in row:
                    results[arm][rel][stage].append(row["mrr"])
                elif isinstance(row, (int, float)):
                    results[arm][rel][stage].append(float(row))
            if rel in ceiling:
                results[arm][rel]["ceiling"].append(ceiling[rel])
        print(f"  {arm} s{seed} done")

def summarise(values):
    """mean +- sample sd; sd is 0 for a single seed rather than undefined."""
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def paired_delta(minuend, subtrahend):
    """Per-seed difference, so the sd is the spread of the DIFFERENCE, not of either term.

    The two stages are evaluated on the same weights at the same seed, so the paired
    spread is the honest one -- propagating the individual sds would ignore that they
    move together and would overstate the uncertainty.
    """
    if not minuend or not subtrahend or len(minuend) != len(subtrahend):
        return float("nan"), float("nan"), 0
    diffs = [a - b for a, b in zip(minuend, subtrahend)]
    mean, sd = summarise(diffs)
    return mean, sd, sum(1 for d in diffs if d > 0)


print(f"\n{'=' * 118}")
print("FLOOR / CASCADE / CEILING -- LOO MRR per inferred relation, mean +- sd across seeds")
print("floor = backbone-only context | cascade = backbone + earlier relations' gold | ceiling = full graph")
print("deltas are paired per seed, so their sd is the spread of the difference")
for arm in arm_names:
    if arm not in results:
        continue
    n = len(seeds)
    print(f"\n{arm}  (n={n} seed{'s' if n > 1 else ''}: {', '.join(str(s) for s in seeds)})")
    print(f"  {'relation':<17}{'floor':>17}{'cascade':>17}{'ceiling':>17}"
          f"{'cascade-floor':>24}{'headroom':>24}")
    for rel in ("MANAGED_FOR", "CONFIRMS", "COMPLICATED_BY", "INDICATES"):
        stage = {s: results[arm][rel][s] for s in ("floor", "cascade", "ceiling")}
        cells = {s: summarise(v) for s, v in stage.items()}
        cg_m, cg_sd, cg_w = paired_delta(stage["cascade"], stage["floor"])
        hr_m, hr_sd, hr_w = paired_delta(stage["ceiling"], stage["floor"])
        flag = "" if hr_m >= 0 else "  <- NEGATIVE"
        floor_s = "{:.4f}+-{:.4f}".format(*cells["floor"])
        casc_s = "{:.4f}+-{:.4f}".format(*cells["cascade"])
        ceil_s = "{:.4f}+-{:.4f}".format(*cells["ceiling"])
        cg_s = "{:+.4f}+-{:.4f} {}/{}".format(cg_m, cg_sd, cg_w, n)
        hr_s = "{:+.4f}+-{:.4f} {}/{}".format(hr_m, hr_sd, hr_w, n)
        print(f"  {rel:<17}{floor_s:>17}{casc_s:>17}{ceil_s:>17}{cg_s:>24}{hr_s:>24}{flag}")

print("\nheadroom = ceiling - floor, what the surrounding context is worth. v18 measured this")
print("for its no-note arms only, at one seed, and found it negative for COMPLICATED_BY and")
print("INDICATES under random masking -- the finding patch masking was introduced to fix.")
