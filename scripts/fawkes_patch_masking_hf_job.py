#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "huggingface_hub>=1.25.1",
#   "numpy>=1.26",
#   "scikit-learn>=1.3",
#   "safetensors>=0.4.5",
#   "torch==2.4.1",
#   "torch-geometric==2.6.1",
# ]
# ///
"""HF Jobs payload for the Fawkes patch-masking experiment.

Submit this file directly with the Hugging Face Jobs CLI.

Recommended, with a recent ``hf`` CLI that supports local directory volumes:

    hf jobs uv run --flavor t4-small --timeout 4h --secrets HF_TOKEN \\
      -v ./src:/workspace/src \\
      scripts/fawkes_patch_masking_hf_job.py

If your CLI says ``Invalid HF URI: ./src:/workspace/src``, it is too old for
local directory volumes. Use this repo's venv CLI:

    .venv/bin/hf jobs uv run --flavor t4-small --timeout 4h --secrets HF_TOKEN \\
      -v ./src:/workspace/src \\
      scripts/fawkes_patch_masking_hf_job.py

or upgrade ``huggingface_hub`` and rerun the ``hf jobs`` command.

Cheap smoke run:

    hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \\
      -v ./src:/workspace/src \\
      -e PUSH=0 -e JEPA_EPOCHS=2 -e READOUT_EPOCHS=2 -e LOO_CAP=200 -e RUN_CASCADE=0 \\
      scripts/fawkes_patch_masking_hf_job.py

This script does not clone git and does not download a code snapshot. The
``-v ./src:/workspace/src`` mount is the source of truth for the job code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


DEFAULT_SOURCE_ROOT = Path("/workspace/src")

DEFAULTS = {
    # v16 entity-note defaults, made explicit for the job log.
    "USE_NOTE": "1",
    "GROUND_BY": "prov",
    "EMBED_DIM": "768",
    "USE_SCORES": "0",
    "PRUNE_NO_EVIDENCE": "1",
    # v17 patch masking experiment.
    "MASK_STRATEGY": "patch",
    "JEPA_PATCHES": "8",
    "NODE_MASK": "0.4",
    # Normal Fawkes evaluation defaults; override for smoke runs.
    "RUN_CASCADE": "1",
    "RUN_EIR": "0",
    "PUSH": "1",
}


def truthy(value: str | None, default: str = "0") -> bool:
    return str(value if value is not None else default).lower() in ("1", "true", "yes")


def hub_username() -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().whoami()["name"]
    except Exception:
        return None


def default_output_repo() -> str | None:
    username = hub_username()
    if not username:
        return None
    stamp = os.environ.get("JOB_ID") or time.strftime("%y%m%d-%H%M%S")
    return f"{username}/fawkes-graph-jepa-v17-patch-{stamp}"


def configure_environment() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    if truthy(os.environ.get("PUSH"), "1") and not os.environ.get("OUTPUT_REPO"):
        repo = default_output_repo()
        if not repo:
            raise SystemExit(
                "PUSH=1 needs OUTPUT_REPO or an HF_TOKEN secret so the job can choose "
                "<user>/fawkes-graph-jepa-v17-patch-<JOB_ID>. Either submit with "
                "`--secrets HF_TOKEN`, pass `-e OUTPUT_REPO=...`, or smoke-test with `-e PUSH=0`."
            )
        os.environ["OUTPUT_REPO"] = repo


def find_source_root(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path(os.environ["FAWKES_SRC"]) if os.environ.get("FAWKES_SRC") else None,
        DEFAULT_SOURCE_ROOT,
        Path(__file__).resolve().parents[1] / "src",
    ]
    for candidate in candidates:
        if candidate and (candidate / "fawkes" / "train.py").is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates if c)
    raise SystemExit(
        "Could not find fawkes/train.py. Submit with `-v ./src:/workspace/src` "
        f"or set FAWKES_SRC. Tried: {tried}"
    )


def log_environment(source_root: Path) -> None:
    keys = sorted(set(DEFAULTS) | {
        "DATA_REPO", "DATA_FILE", "DATA_PATH", "OUTPUT_REPO", "FAWKES_SRC",
        "NOTE_INJECTION", "NOTE_MEMORY_REPO", "NOTE_MEMORY_FILE",
        "NOTE_MEMORY_PATH", "NOTE_SPAN_TOKENS", "NOTE_MAX_SPANS",
        "NOTE_ATTN_HEADS", "DECODER", "DATA_SPLIT_SEED", "SEED",
    })
    print(f"[FAWKES-HF] source_root={source_root}", flush=True)
    print("[FAWKES-HF] no git clone or code snapshot download is performed", flush=True)
    print("[FAWKES-HF] trainer environment:", flush=True)
    for key in keys:
        if key in os.environ:
            print(f"[FAWKES-HF]   {key}={os.environ[key]}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Fawkes patch-masking training/evaluation inside an HF Job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "HF CLI env overrides go before the script path. A recent hf CLI is needed for local -v mounts:\n"
            "  .venv/bin/hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN "
            "-v ./src:/workspace/src -e PUSH=0 -e JEPA_EPOCHS=2 "
            "scripts/fawkes_patch_masking_hf_job.py"
        ),
    )
    parser.add_argument("--source-root", help="source tree containing fawkes/; defaults to /workspace/src")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_environment()
    source_root = find_source_root(args.source_root)
    sys.path.insert(0, str(source_root))
    log_environment(source_root)

    from fawkes.train import main as train_main
    train_main([])


if __name__ == "__main__":
    main()
