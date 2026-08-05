import statistics
import torch
from huggingface_hub import HfApi, hf_hub_download

user = HfApi().whoami()["name"]
runs = [(a, s) for s in (42, 43, 44) for a in ("A", "B", "C", "Cp")] + [("D", 42), ("E", 42)]
rows = {}

print(f"{'run':<10}{'MRR':>9}{'H@1':>9}{'H@3':>9}{'H@10':>9}{'n':>8}")
for arm, seed in runs:
    label = f"{arm}-s{seed}"
    try:
        path = hf_hub_download(f"kushagrayadv/fawkes-v19-ablation-{arm}-s{seed}", "fawkes_entity_note.pt", local_dir="/Users/kushagrayadav/Code/clinical-graph-jepa/data/fawkes_ablations")
    except Exception as exc:
        print(f"{label:<10}  MISSING ({type(exc).__name__}) — check `hf jobs logs`")
        continue
    m = torch.load(path, map_location="cpu", weights_only=False)["recovery_test_loo"]
    rows[(arm, seed)] = m
    print(f"{label:<10}{m['mrr']:>9.4f}{m['hits1']:>9.4f}{m['hits3']:>9.4f}{m['hits10']:>9.4f}{m['n']:>8}")

print("\nper arm, LOO MRR:")
for arm in ("A", "B", "C", "Cp", "D", "E"):
    vals = [rows[(arm, s)]["mrr"] for s in (42, 43, 44) if (arm, s) in rows]
    if not vals:
        continue
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {arm:<4} {statistics.mean(vals):.4f} +/- {sd:.4f}  ({len(vals)} seed(s))")

print("\npaired delta vs A, per seed:")
for arm in ("B", "C", "Cp"):
    for s in (42, 43, 44):
        if (arm, s) in rows and ("A", s) in rows:
            print(f"  A - {arm} @ s{s}: {rows[('A', s)]['mrr'] - rows[(arm, s)]['mrr']:+.4f}")