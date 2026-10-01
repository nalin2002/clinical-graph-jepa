#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "huggingface_hub>=0.34,<1",
#   "numpy>=1.26",
#   "scikit-learn>=1.3",
#   "safetensors>=0.4.5",
#   "torch==2.4.1",
#   "torch-geometric==2.6.1",
#   "tqdm>=4.66",
#   "transformers>=4.44,<5",
# ]
# ///
"""HF Jobs payload for Clinical-JEPA note/no-note seed sweeps."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download


os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")

DEFAULT_SOURCE_ROOT = Path("/workspace/src")
DEFAULT_MODELS_ROOT = Path("/workspace/models")
DEFAULT_DATA_REPO = "wmatbooth/fawkes-training-graph-embedded-260615"
DEFAULT_DATA_FILE = "fawkes_training_graph_full_embedded_260615.jsonl"
DEFAULT_RUN_ROOT = Path("/tmp/clinical-jepa")


def truthy(value: str | None, default: str = "0") -> bool:
    return str(value if value is not None else default).lower() in {
        "1",
        "true",
        "yes",
    }


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def find_source_root(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path(os.environ["CLINICAL_JEPA_SRC"])
        if os.environ.get("CLINICAL_JEPA_SRC")
        else None,
        DEFAULT_SOURCE_ROOT,
        Path(__file__).resolve().parents[1] / "src",
    ]
    for candidate in candidates:
        if candidate and (candidate / "clinical_jepa" / "train" / "pretrain.py").is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates if candidate)
    raise SystemExit(
        "Could not find clinical_jepa/train/pretrain.py. Submit with "
        "`-v ./src:/workspace/src` or set CLINICAL_JEPA_SRC. Tried: "
        f"{tried}"
    )


def find_models_root(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path(os.environ["CLINICAL_JEPA_MODELS"])
        if os.environ.get("CLINICAL_JEPA_MODELS")
        else None,
        DEFAULT_MODELS_ROOT,
        Path(__file__).resolve().parents[1] / "models",
    ]
    for candidate in candidates:
        if candidate and (
            candidate
            / "fawkes-entity-note"
            / "fawkes_trainer_jepa_entity_note_v16_260615.pt"
        ).is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates if candidate)
    raise SystemExit(
        "Could not find the Fawkes checkpoint. Submit with "
        "`-v ./models:/workspace/models` or set CLINICAL_JEPA_MODELS. Tried: "
        f"{tried}"
    )


def hub_username() -> str | None:
    try:
        return HfApi().whoami()["name"]
    except Exception:
        return None


def default_output_repo(variant: str, split_seed: int, seed: int) -> str | None:
    username = hub_username()
    if not username:
        return None
    stamp = os.environ.get("JOB_ID") or time.strftime("%y%m%d-%H%M%S")
    return (
        f"{username}/clinical-jepa-{variant}-"
        f"fawkes-split-sp{split_seed}-s{seed}-{stamp}"
    )


def configure_output_repo(variant: str, split_seed: int, seed: int) -> str | None:
    if not truthy(os.environ.get("PUSH"), "1"):
        return None
    repo = os.environ.get("OUTPUT_REPO") or default_output_repo(
        variant,
        split_seed,
        seed,
    )
    if not repo:
        raise SystemExit(
            "PUSH=1 needs OUTPUT_REPO or an HF_TOKEN secret so the job can choose "
            "<user>/clinical-jepa-<variant>-fawkes-split-... Either submit with "
            "`--secrets HF_TOKEN`, pass `-e OUTPUT_REPO=...`, or smoke-test with "
            "`-e PUSH=0`."
        )
    return repo


def checkpoint_name(variant: str, *, pretrain: bool) -> str:
    variant_token = variant.replace("-", "_")
    stage = "_pretrain" if pretrain else ""
    return f"clinical_jepa_{variant_token}{stage}.pt"


def hf_resume_enabled(repo_id: str | None) -> bool:
    return bool(repo_id) and truthy(os.environ.get("RESUME_FROM_HF"), "1")


def missing_hf_file(exc: Exception) -> bool:
    if type(exc).__name__ in {
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "RepositoryNotFoundError",
    }:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def expected_checkpoint_config(
    *,
    variant: str,
    stage: str,
    seed: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    lr: float,
    encoder: str,
) -> dict:
    expected = {
        "encoder": encoder,
        "use_note_embeddings": variant == "note",
        "seed": seed,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    if stage == "pretrain":
        expected["finetune_epochs"] = 0
    else:
        expected["finetune_epochs"] = finetune_epochs
        expected["llm_confidence_negatives"] = truthy(
            os.environ.get("LLM_CONFIDENCE_NEGATIVES"),
            "1",
        )
        expected["clinical_artifact_filters"] = truthy(
            os.environ.get("CLINICAL_ARTIFACT_FILTERS"),
            "1",
        )
        expected["llm_negative_weight"] = env_float("LLM_NEGATIVE_WEIGHT", 0.6)
    return expected


def checkpoint_matches(path: Path, expected: dict) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
    except Exception as exc:
        print(
            f"[RESUME] {path.name} exists but cannot be inspected "
            f"({type(exc).__name__}); retraining",
            flush=True,
        )
        return False

    train = cfg.get("train", {})
    model = cfg.get("model", {})
    actual = {
        "encoder": cfg.get("encoder"),
        "use_note_embeddings": bool(model.get("use_note_embeddings")),
        "seed": train.get("seed"),
        "pretrain_epochs": train.get("pretrain_epochs"),
        "finetune_epochs": train.get("finetune_epochs"),
        "batch_size": train.get("batch_size"),
        "lr": train.get("lr"),
        "llm_confidence_negatives": train.get("llm_confidence_negatives"),
        "clinical_artifact_filters": train.get("clinical_artifact_filters"),
        "llm_negative_weight": train.get("llm_negative_weight"),
    }

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or not math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
        elif actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

    if mismatches:
        print(
            f"[RESUME] {path.name} config mismatch; retraining "
            f"({'; '.join(mismatches)})",
            flush=True,
        )
        return False
    return True


def maybe_download_checkpoint(
    repo_id: str | None,
    filename: str,
    out_dir: Path,
    *,
    expected: dict,
) -> Path | None:
    if not hf_resume_enabled(repo_id):
        return None
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(out_dir),
        )
    except Exception as exc:
        if missing_hf_file(exc):
            print(f"[RESUME] miss hf://models/{repo_id}/{filename}", flush=True)
            return None
        raise

    path = Path(local)
    print(f"[RESUME] found hf://models/{repo_id}/{filename}", flush=True)
    if not checkpoint_matches(path, expected):
        return None
    print(f"[RESUME] using {path}; skipping matching stage", flush=True)
    return path


def upload_checkpoint_artifacts(
    repo_id: str | None,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    config_name: str,
) -> None:
    if not repo_id:
        print(f"[UPLOAD] PUSH=0, not uploading {checkpoint_path.name}", flush=True)
        return
    api = HfApi()
    private = truthy(os.environ.get("PRIVATE"), "1")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    config_path = out_dir / config_name
    if config_path.is_file():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=config_name,
            repo_id=repo_id,
            repo_type="model",
        )
    print(
        f"[UPLOAD] checkpoint https://huggingface.co/{repo_id}/blob/main/"
        f"{checkpoint_path.name}",
        flush=True,
    )


def upload_file_with_retries(
    api: HfApi,
    *,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    attempts: int = 3,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            return
        except Exception:
            if attempt == attempts:
                raise
            sleep_s = 2 * attempt
            print(
                f"[UPLOAD] retry {attempt}/{attempts} for {path_in_repo} "
                f"after {sleep_s}s",
                flush=True,
            )
            time.sleep(sleep_s)


def maybe_limit_data(data_path: Path, run_root: Path) -> Path:
    raw_limit = os.environ.get("DATA_LIMIT")
    if not raw_limit:
        return data_path
    limit = int(raw_limit)
    if limit <= 0:
        raise SystemExit("DATA_LIMIT must be positive")

    limited_dir = run_root / "data"
    limited_dir.mkdir(parents=True, exist_ok=True)
    limited_path = limited_dir / f"{data_path.stem}.first{limit}{data_path.suffix}"
    copied = 0
    with data_path.open(encoding="utf-8") as source, limited_path.open(
        "w",
        encoding="utf-8",
    ) as dest:
        for copied, line in enumerate(source, start=1):
            if copied > limit:
                copied -= 1
                break
            dest.write(line)
    if copied == 0:
        raise SystemExit(f"DATA_LIMIT produced an empty file from {data_path}")
    print(
        f"[DATA] limited_records={copied} limit={limit} local={limited_path}",
        flush=True,
    )
    return limited_path


def download_data(run_root: Path) -> Path:
    if os.environ.get("DATA_PATH"):
        data_path = Path(os.environ["DATA_PATH"])
        if not data_path.is_file():
            raise SystemExit(f"DATA_PATH does not exist: {data_path}")
        print(f"[DATA] local DATA_PATH={data_path}", flush=True)
        return maybe_limit_data(data_path, run_root)

    data_repo = os.environ.get("DATA_REPO", DEFAULT_DATA_REPO)
    data_file = os.environ.get("DATA_FILE", DEFAULT_DATA_FILE)
    local = hf_hub_download(data_repo, data_file, repo_type="dataset")
    data_dir = run_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / data_file
    data_path.write_bytes(Path(local).read_bytes())
    print(f"[DATA] hf://datasets/{data_repo}/{data_file}", flush=True)
    print(f"[DATA] local={data_path}", flush=True)
    return maybe_limit_data(data_path, run_root)


def write_fawkes_train_plus_val_split(
    data_path: Path,
    split_path: Path,
    *,
    split_seed: int,
) -> dict:
    from fawkes.config import Config as FawkesConfig
    from fawkes.data import to_data
    from fawkes.evaluate import _load_graphs

    raw, demographics = _load_graphs(data_path, None)
    cfg = replace(
        FawkesConfig(),
        seed=split_seed,
        data_split_seed=split_seed,
    )
    eligible: list[int] = []
    for index, graph in enumerate(raw):
        data = to_data(graph, demographics, cfg)
        if data.num_nodes >= 3 and data.edge_index.size(1) >= 4:
            eligible.append(index)

    order = np.random.RandomState(cfg.data_split_seed).permutation(len(eligible))
    num_test = int(cfg.test_frac * len(eligible))
    test = {eligible[index] for index in order[:num_test]}
    train_plus_val = set(eligible) - test

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as stream:
        for index, graph in enumerate(raw):
            if index in train_plus_val:
                stream.write(json.dumps(graph) + "\n")

    manifest = {
        "records": len(raw),
        "eligible": len(eligible),
        "split_seed": cfg.data_split_seed,
        "test_frac": cfg.test_frac,
        "val_frac": cfg.val_frac,
        "train_plus_val": len(train_plus_val),
        "test": len(test),
        "split_path": str(split_path),
    }
    print(f"[SPLIT] {json.dumps(manifest, sort_keys=True)}", flush=True)
    return manifest


def choose_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_transformers_torch_backend() -> None:
    import transformers
    from transformers.utils import import_utils

    torch_available = import_utils.is_torch_available()
    print(
        "[RUNTIME] "
        f"python={sys.version.split()[0]} "
        f"torch={torch.__version__} "
        f"transformers={transformers.__version__} "
        f"transformers_torch_available={torch_available} "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )
    if not torch_available:
        raise SystemExit(
            "transformers does not see PyTorch. Check USE_TORCH/USE_TF and the "
            "torch/transformers package pins in the HF job environment."
        )


def train_variant(
    source_root: Path,
    train_path: Path,
    out_dir: Path,
    *,
    variant: str,
    seed: int,
    device: str,
    output_repo: str | None,
) -> Path:
    sys.path.insert(0, str(source_root))
    from clinical_jepa.train import finetune, pretrain

    pretrain_epochs = env_int("PRETRAIN_EPOCHS", 60)
    if os.environ.get("FINETUNE_EPOCHS"):
        finetune_epochs = env_int("FINETUNE_EPOCHS", 0)
    else:
        finetune_epochs = 50 if variant == "note" else 90
    batch_size = env_int("BATCH_SIZE", 16)
    lr = env_float("LR", 8e-4)
    encoder = os.environ.get("ENCODER", "sapbert")
    encoder_cache = os.environ.get("ENCODER_CACHE", str(out_dir / "encoder-cache"))
    pretrain_checkpoint_name = checkpoint_name(variant, pretrain=True)
    finetune_checkpoint_name = checkpoint_name(variant, pretrain=False)
    pretrain_expected = expected_checkpoint_config(
        variant=variant,
        stage="pretrain",
        seed=seed,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        lr=lr,
        encoder=encoder,
    )
    finetune_expected = expected_checkpoint_config(
        variant=variant,
        stage="finetune",
        seed=seed,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        lr=lr,
        encoder=encoder,
    )

    finetune_path = maybe_download_checkpoint(
        output_repo,
        finetune_checkpoint_name,
        out_dir,
        expected=finetune_expected,
    )
    if finetune_path is not None:
        return finetune_path

    pretrain_argv = [
        "--data",
        "jsonl",
        "--jsonl-path",
        str(train_path),
        "--out",
        str(out_dir),
        "--encoder",
        encoder,
        "--encoder-cache",
        encoder_cache,
        "--epochs",
        str(pretrain_epochs),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
    ]
    if variant == "no-note":
        pretrain_argv.append("--no-note-embeddings")
    elif variant != "note":
        raise SystemExit("VARIANT must be 'note' or 'no-note'")

    pretrain_path = maybe_download_checkpoint(
        output_repo,
        pretrain_checkpoint_name,
        out_dir,
        expected=pretrain_expected,
    )
    if pretrain_path is None:
        print(f"[TRAIN] pretrain argv={pretrain_argv}", flush=True)
        pretrain_path = pretrain.pretrain(
            pretrain.build_arg_parser().parse_args(pretrain_argv)
        )
        upload_checkpoint_artifacts(
            output_repo,
            pretrain_path,
            out_dir,
            config_name="config_pretrain.json",
        )

    finetune_argv = [
        "--data",
        "jsonl",
        "--jsonl-path",
        str(train_path),
        "--checkpoint",
        str(pretrain_path),
        "--out",
        str(out_dir),
        "--encoder-cache",
        encoder_cache,
        "--epochs",
        str(finetune_epochs),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
    ]
    if truthy(os.environ.get("LLM_CONFIDENCE_NEGATIVES"), "1"):
        finetune_argv.append("--llm-confidence-negatives")
    if truthy(os.environ.get("CLINICAL_ARTIFACT_FILTERS"), "1"):
        finetune_argv.append("--clinical-artifact-filters")
    finetune_argv += [
        "--llm-negative-weight",
        os.environ.get("LLM_NEGATIVE_WEIGHT", "0.6"),
    ]

    print(f"[TRAIN] finetune argv={finetune_argv}", flush=True)
    finetune_path = finetune.finetune(
        finetune.build_arg_parser().parse_args(finetune_argv)
    )
    upload_checkpoint_artifacts(
        output_repo,
        finetune_path,
        out_dir,
        config_name="config.json",
    )
    return finetune_path


def evaluate_shared_queries(
    source_root: Path,
    models_root: Path,
    data_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
) -> dict:
    sys.path.insert(0, str(source_root))
    from benchmarks import shared_queries

    fawkes_checkpoint = (
        models_root
        / "fawkes-entity-note"
        / "fawkes_trainer_jepa_entity_note_v16_260615.pt"
    )
    output_path = out_dir / "shared_queries_eval.json"
    args = shared_queries.build_arg_parser().parse_args(
        [
            "--data",
            str(data_path),
            "--fawkes-checkpoint",
            str(fawkes_checkpoint),
            "--checkpoint",
            str(checkpoint_path),
            "--encoder-cache",
            os.environ.get("EVAL_ENCODER_CACHE", str(out_dir / "eval-encoder-cache")),
            "--device",
            device,
            "--cap",
            os.environ.get("EVAL_CAP", "40000"),
            "--output",
            str(output_path),
        ]
    )
    payload = shared_queries.run(args)
    print(f"[EVAL] wrote {output_path}", flush=True)
    return payload


def write_readme(out_dir: Path, config: dict, metrics: dict) -> None:
    clinical = metrics["clinical_jepa"]
    text = f"""# Clinical-JEPA seed run

```json
{json.dumps(config, indent=2, sort_keys=True)}
```

Shared-query Clinical-JEPA metrics:

| MRR | H@1 | H@3 | H@10 | n |
| ---: | ---: | ---: | ---: | ---: |
| {clinical['mrr']:.6f} | {clinical['hits1']:.6f} | {clinical['hits3']:.6f} | {clinical['hits10']:.6f} | {clinical['n']} |
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def maybe_upload(out_dir: Path, repo_id: str | None) -> None:
    if not repo_id:
        print("[UPLOAD] PUSH=0, not uploading", flush=True)
        return
    api = HfApi()
    private = truthy(os.environ.get("PRIVATE"), "1")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    uploaded = []
    for name in (
        "README.md",
        "job_config.json",
        "shared_queries_eval.json",
        "config.json",
        "config_pretrain.json",
        "clinical_jepa_note.pt",
        "clinical_jepa_note_pretrain.pt",
        "clinical_jepa_no_note.pt",
        "clinical_jepa_no_note_pretrain.pt",
    ):
        path = out_dir / name
        if not path.is_file():
            continue
        upload_file_with_retries(
            api,
            repo_id=repo_id,
            local_path=path,
            path_in_repo=name,
        )
        uploaded.append(name)
    print(f"[UPLOAD] artifacts={uploaded}", flush=True)
    print(f"[UPLOAD] https://huggingface.co/{repo_id}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Mounted src directory")
    parser.add_argument("--models-root", help="Mounted models directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_root = find_source_root(args.source_root)
    models_root = find_models_root(args.models_root)
    sys.path.insert(0, str(source_root))
    check_transformers_torch_backend()

    variant = os.environ.get("VARIANT", "note").replace("_", "-")
    seed = env_int("SEED", 42)
    split_seed = env_int("DATA_SPLIT_SEED", 42)
    device = choose_device()
    output_repo = configure_output_repo(variant, split_seed, seed)
    run_root = Path(os.environ.get("RUN_ROOT", DEFAULT_RUN_ROOT))
    out_dir = run_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "variant": variant,
        "seed": seed,
        "data_split_seed": split_seed,
        "device": device,
        "output_repo": output_repo,
        "source_root": str(source_root),
        "models_root": str(models_root),
        "data_limit": os.environ.get("DATA_LIMIT"),
    }
    print(f"[CONFIG] {json.dumps(config, sort_keys=True)}", flush=True)
    data_path = download_data(run_root)
    split_path = run_root / "splits" / "fawkes_train_plus_val.jsonl"
    split_manifest = write_fawkes_train_plus_val_split(
        data_path,
        split_path,
        split_seed=split_seed,
    )
    checkpoint_path = train_variant(
        source_root,
        split_path,
        out_dir,
        variant=variant,
        seed=seed,
        device=device,
        output_repo=output_repo,
    )
    metrics = evaluate_shared_queries(
        source_root,
        models_root,
        data_path,
        checkpoint_path,
        out_dir,
        device=device,
    )
    config["split"] = split_manifest
    config_path = out_dir / "job_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(out_dir, config, metrics)
    maybe_upload(out_dir, output_repo)


if __name__ == "__main__":
    main()
