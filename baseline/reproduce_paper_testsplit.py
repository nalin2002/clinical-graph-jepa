"""Reproduce the trainer's seeded TEST split and re-run loo_evaluate on it.

One-off Phase 0 verification: does the checkpoint on disk reproduce the
recovery_test_loo numbers stored inside it? Mirrors trainer.py:600-610 exactly,
including the >=4 edge filter (evaluate.py uses >=2, which is a different set).
"""
import json, sys
import numpy as np, torch

sys.path.insert(0, "src")
from paper_v16 import trainer
from paper_v16.evaluate import _load_graphs
from pathlib import Path

CKPT = "models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt"
DATA = "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
raw, demo = _load_graphs(Path(DATA), None)

items = []
for g in raw:
    d = trainer.to_data(g, demo)
    if d.num_nodes >= 3 and d.edge_index.size(1) >= 4:      # trainer.py:601
        items.append(d)

rng = np.random.RandomState(trainer.SEED)
idx = rng.permutation(len(items))
nte = int(trainer.TEST_FRAC * len(items))
te = [items[i] for i in idx[:nte]]
print(f"items={len(items)} TEST={len(te)} seed={trainer.SEED} test_frac={trainer.TEST_FRAC}")

dev = torch.device("cpu")
enc, sc = trainer.Encoder().to(dev), trainer.DistMult().to(dev)
enc.load_state_dict(ck["encoder"]); sc.load_state_dict(ck["scorer"])
m = trainer.loo_evaluate(enc, sc, te, dev, cap=trainer.LOO_CAP)

emb = ck["recovery_test_loo"]
print()
for k in ("mrr", "hits1", "hits3", "hits10", "n"):
    a, b = m.get(k), emb.get(k)
    if isinstance(a, float):
        print(f"{k:8} rerun={a:.9f}  checkpoint={b:.9f}  delta={abs(a-b):.3e}")
    else:
        print(f"{k:8} rerun={a}  checkpoint={b}  {'MATCH' if a==b else 'DIFFER'}")
json.dump(m, open(f"{sys.argv[1]}", "w"), indent=2)
