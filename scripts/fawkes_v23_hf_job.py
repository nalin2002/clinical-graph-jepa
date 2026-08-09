#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "huggingface_hub>=1.25.1",
#   "numpy>=1.26",
#   "safetensors>=0.4.5",
#   "scikit-learn>=1.3",
#   "torch==2.4.1",
#   "torch-geometric==2.6.1",
# ]
# ///
"""Reproducible HF Jobs entry point for the Fawkes v23 note-injection arms."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tarfile

from huggingface_hub import hf_hub_download


DEFAULTS = {
    "USE_NOTE": "1",
    "GROUND_BY": "prov",
    "EMBED_DIM": "768",
    "USE_SCORES": "0",
    "PRUNE_NO_EVIDENCE": "1",
    "MASK_STRATEGY": "patch",
    "JEPA_PATCHES": "8",
    "NODE_MASK": "0.4",
    "DECODER": "mlp",
    "NOTE_INJECTION": "attention",
    "NOTE_MEMORY_REPO": "nalin9/fawkes-training-note-memory-v23-260808",
    "NOTE_MEMORY_FILE": "fawkes_note_memory_v23.safetensors",
    "NOTE_SPAN_TOKENS": "32",
    "NOTE_MAX_SPANS": "64",
    "NOTE_ATTN_HEADS": "4",
    "RUN_CASCADE": "0",
    "RUN_EIR": "0",
    "PUSH": "1",
    "DATA_SPLIT_SEED": "42",
}


def configure() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    required = ["SOURCE_REF", "OUTPUT_REPO"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"missing required environment variables: {missing}")


def fetch_source() -> Path:
    repository = os.environ.get("SOURCE_REPO", "nalin9/clinical-graph-jepa-v23-source")
    revision = os.environ["SOURCE_REF"]
    filename = f"{revision}.tar.gz"
    archive_path = hf_hub_download(repository, filename, repo_type="dataset")
    print(f"[SOURCE] hf://datasets/{repository}/{filename}", flush=True)
    root = Path("/tmp/fawkes-v23-source")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if root.resolve() not in target.parents and target != root.resolve():
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(root)
    source_dirs = [path for path in [root / "src", *root.glob("*/src")] if path.is_dir()]
    if len(source_dirs) != 1 or not (source_dirs[0] / "fawkes" / "train.py").is_file():
        raise RuntimeError(f"could not resolve fawkes source under {root}")
    return source_dirs[0]


def main() -> None:
    configure()
    source_root = fetch_source()
    sys.path.insert(0, str(source_root))
    print(f"[SOURCE] source_root={source_root}", flush=True)
    for key in sorted(set(DEFAULTS) | {"SOURCE_REF", "SOURCE_REPO", "OUTPUT_REPO", "SEED"}):
        if key in os.environ:
            print(f"[CONFIG] {key}={os.environ[key]}", flush=True)
    from fawkes.train import main as train_main
    train_main([])


if __name__ == "__main__":
    main()
