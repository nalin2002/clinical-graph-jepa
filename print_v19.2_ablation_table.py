"""v19.2 — v19's JEPA-pretraining ablation, read off the v22 configuration.

Arm A is not a v19.2 run. Its environment is v22's exactly (every ablated knob at its
default) and DETERMINISTIC=1, so the v22 sp42 runs ARE arm A and are read from their
own repos. Every arm shares DATA_SPLIT_SEED=42, so each comparison against A is paired:
same held-out admissions, same LOO edges, same denominator.
"""
import statistics
import torch
from huggingface_hub import hf_hub_download

CACHE = "/Users/kushagrayadav/Code/clinical-graph-jepa/data/fawkes_v19.2"

# arm -> (seeds, what it changes vs A)
ARMS = {
    "A":  ((42, 43, 44), "JEPA 60 ep, frozen encoder (= v22)"),
    "B":  ((42, 43, 44), "no phase 1, frozen random init"),
    "C":  ((42, 43, 44), "no phase 1, encoder trained jointly"),
    "Cp": ((42, 43, 44), "as C, READOUT_EPOCHS=100"),
    "D":  ((42,),        "as C, LAYERS=0 — no message passing"),
    "E":  ((42,),        "JEPA 60 ep, encoder fine-tuned"),
}


def repo(arm, seed):
    if arm == "A":
        return f"kushagrayadv/fawkes-v22-patch-mlp-sp42-s{seed}"
    # v19-2, not v19.2: HF job names are stored as tags and tags reject '.', so the
    # submit script drops it and the repos follow. See submit-v19.2-ablations.sh.
    return f"kushagrayadv/fawkes-v19-2-ablation-{arm}-sp42-s{seed}"


rows = {}
print(f"{'run':<10}{'MRR':>9}{'H@1':>9}{'H@3':>9}{'H@10':>9}{'n':>8}")
for arm, (seeds, _what) in ARMS.items():
    for seed in seeds:
        label = f"{arm}-s{seed}"
        try:
            path = hf_hub_download(repo(arm, seed), "fawkes_entity_note.pt",
                                   local_dir=f"{CACHE}/{label}")
        except Exception as exc:
            print(f"{label:<10}  MISSING ({type(exc).__name__}) — check `hf jobs logs`")
            continue
        m = torch.load(path, map_location="cpu", weights_only=False)["recovery_test_loo"]
        rows[(arm, seed)] = m
        print(f"{label:<10}{m['mrr']:>9.4f}{m['hits1']:>9.4f}{m['hits3']:>9.4f}{m['hits10']:>9.4f}{m['n']:>8}")

print("\nper arm, LOO MRR:")
for arm, (seeds, what) in ARMS.items():
    vals = [rows[(arm, s)]["mrr"] for s in seeds if (arm, s) in rows]
    if not vals:
        continue
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {arm:<4} {statistics.mean(vals):.4f} +/- {sd:.4f}  ({len(vals)} seed(s))  {what}")

print("\npaired delta vs A, per seed:")
for arm, (seeds, _what) in ARMS.items():
    if arm == "A":
        continue
    paired = [(s, rows[(arm, s)]["mrr"] - rows[("A", s)]["mrr"])
              for s in seeds if (arm, s) in rows and ("A", s) in rows]
    if not paired:
        continue
    for s, d in paired:
        print(f"  {arm} - A @ s{s}: {d:+.4f}")
    diffs = [d for _s, d in paired]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    # sd over 3 seeds, not a CI: t(2)=4.303 makes an interval too wide to decide anything.
    # Read the sign consistency across seeds, as the v19 report did.
    print(f"    mean {statistics.mean(diffs):+.4f} +/- {sd:.4f}  "
          f"positive at {sum(1 for d in diffs if d > 0)}/{len(diffs)}")

print("\nDECODER is not written into checkpoint_dict(), so these files cannot identify")
print("their own head — the repo name is the provenance record. Re-evaluating any of")
print("them needs DECODER=mlp set, arm A included.")
