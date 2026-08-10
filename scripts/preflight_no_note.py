"""Local pre-flight for the no-note arms of v20/v21/v22, before any GPU is spent.

The run-rule this repo learned the hard way: validate locally before submitting,
because a broken arm is only visible after the jobs have run and been paid for.

Every v20/v21/v22 run so far used USE_NOTE=1. The no-note variant halves the node
input (numeric_dim 774 -> 6, so the encoder's input projection is a different
shape), changes the parameter count, and writes a different checkpoint filename.
Two of the three combinations below — no-note with the MLP head, and no-note with
patch masking AND the MLP head — have never been executed anywhere in this repo.

This runs all three end to end on a small sample: both training phases, the
batch-mask evaluator, and leave-one-out. It asserts nothing about quality; it only
proves the code paths run and the artifacts describe themselves correctly.
"""
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from fawkes.config import Config
from fawkes.evaluate import _load_graphs, evaluate, loo_evaluate
from fawkes.data import to_data
from fawkes.model import JEPA, build_scorer
from fawkes.steps import jepa_step, readout_step

DATA = "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"
N_GRAPHS = 64

# (label, overrides) — the three arms whose no-note cells are missing.
ARMS = [
    ("v20-nonote  random  distmult", dict(mask_strategy="random", decoder="distmult")),
    ("v21-nonote  random  mlp     ", dict(mask_strategy="random", decoder="mlp")),
    ("v22-nonote  patch   mlp     ", dict(mask_strategy="patch", decoder="mlp", jepa_patches=8)),
]

device = torch.device("cpu")
failures = []

# Load once with a no-note config; to_data drops the note dimensions, so the same
# converted graphs serve every arm below.
base = Config(data_path=DATA, use_note=False)
print(f"numeric_dim = {base.numeric_dim}  (must be 6; the note variant is 774)")
if base.numeric_dim != 6:
    sys.exit(f"[FAIL] no-note numeric_dim is {base.numeric_dim}, expected 6")

raw, demographics = _load_graphs(Path(DATA), N_GRAPHS)
graphs = [d for g in raw
          if (d := to_data(g, demographics, base)).num_nodes >= 3 and d.edge_index.size(1) >= 4]
print(f"sample = {len(graphs)} graphs, node feature width = {graphs[0].numfeat.size(1)}\n")

for label, overrides in ARMS:
    cfg = Config(data_path=DATA, use_note=False, push=False, **overrides)
    started = time.time()
    try:
        torch.manual_seed(42)
        model = JEPA(cfg).to(device)
        scorer = build_scorer(cfg).to(device)
        loader = DataLoader(graphs, batch_size=16, shuffle=False)
        batch = next(iter(loader))

        # Phase 1 — the pretext task, which is what mask_strategy switches.
        loss, emb_std = jepa_step(model, batch, device, cfg)
        if not torch.isfinite(loss):
            raise ValueError(f"phase-1 loss is {loss}")

        # Phase 2 — the readout, which is what decoder switches.
        step_out = readout_step(model.ctx, scorer, batch, device, True, cfg)
        if step_out is None:
            raise ValueError("readout_step returned None on a 16-graph batch")

        # Both evaluators, including the metric corrected on 260809.
        eval_loader = DataLoader(graphs, batch_size=1, shuffle=False)
        batchmask = evaluate(model.ctx, scorer, eval_loader, device, cfg)
        loo = loo_evaluate(model.ctx, scorer, graphs, device, cfg, cap=200)

        params = sum(p.numel() for p in model.ctx.parameters())
        run_config = cfg.run_dict()
        assert cfg.checkpoint_name == "fawkes_no_note.pt", cfg.checkpoint_name
        assert run_config["decoder"] == overrides["decoder"]
        assert run_config["mask_strategy"] == overrides["mask_strategy"]

        print(f"[OK]   {label}  encoder params={params:,}  jepa_loss={loss:.4f} emb_std={emb_std:.3f}")
        print(f"       batch-mask auc={batchmask['auc']:.4f} nonobv={batchmask['auc_nonobvious']:.4f}"
              f"  |  LOO mrr={loo['mrr']:.4f} n={loo['n']}  |  {time.time() - started:.1f}s")
        print(f"       checkpoint={cfg.checkpoint_name}  run_config={run_config}\n")
    except Exception as exc:
        failures.append((label, exc))
        print(f"[FAIL] {label}  {type(exc).__name__}: {exc}\n")

if failures:
    sys.exit(f"[FAIL] {len(failures)} of {len(ARMS)} arms did not run — do not submit")
print("All three no-note arms run end to end. Safe to submit.")
