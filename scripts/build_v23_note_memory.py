#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "huggingface_hub>=0.34,<1",
#   "safetensors>=0.4.5",
#   "torch==2.4.1",
#   "transformers>=4.48,<5",
# ]
# ///
"""Build the fixed Clinical ModernBERT span-memory sidecar used by Fawkes v23."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer


DATA_REPO = os.environ.get("DATA_REPO", "wmatbooth/fawkes-training-graph-embedded-260615")
DATA_FILE = os.environ.get("DATA_FILE", "fawkes_training_graph_full_embedded_260615.jsonl")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "wmatbooth/fawkes-training-note-memory-v23-260808")
MODEL_ID = os.environ.get("EMBED_MODEL", "Simonlee711/Clinical_ModernBERT")
MODEL_REVISION = os.environ.get(
    "EMBED_REVISION", "24e72d609fd5dec4607714eed2556235fea5f0a3")
SPAN_TOKENS = int(os.environ.get("NOTE_SPAN_TOKENS", 32))
MAX_SPANS = int(os.environ.get("NOTE_MAX_SPANS", 64))
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", 8))
MIN_MEAN_COSINE = float(os.environ.get("MIN_MEAN_COSINE", 0.99))
OUTPUT_FILE = "fawkes_note_memory_v23.safetensors"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("note"):
                raise RuntimeError(f"record {line_number} has no discharge note")
            if not record.get("note_embedding"):
                raise RuntimeError(f"record {line_number} has no v22 mean embedding")
            if record.get("hadm_id") is None:
                raise RuntimeError(f"record {line_number} has no hadm_id")
            records.append(record)
    if not records:
        raise RuntimeError("source dataset contains zero records")
    return records


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to read and publish the private artifact")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for Clinical ModernBERT preprocessing")

    source = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset")
    records = load_records(source)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16).eval().cuda()
    hidden_size = int(model.config.hidden_size)
    if hidden_size != 768:
        raise RuntimeError(f"Clinical ModernBERT hidden size {hidden_size} != 768")

    memory = torch.zeros((len(records), MAX_SPANS, hidden_size), dtype=torch.float16)
    mask = torch.zeros((len(records), MAX_SPANS), dtype=torch.bool)
    counts = torch.zeros((len(records), MAX_SPANS), dtype=torch.int16)
    hadm_ids = torch.tensor([int(record["hadm_id"]) for record in records], dtype=torch.int64)
    if hadm_ids.unique().numel() != hadm_ids.numel():
        raise RuntimeError("source dataset contains duplicate hadm_id values")

    cosine_values = []
    token_lengths = []
    with torch.inference_mode():
        for offset in range(0, len(records), BATCH_SIZE):
            batch_records = records[offset:offset + BATCH_SIZE]
            encoded = tokenizer(
                [record["note"] for record in batch_records], padding=True,
                truncation=False, return_tensors="pt")
            if encoded["input_ids"].size(1) > int(model.config.max_position_embeddings):
                raise RuntimeError(
                    f"batch at row {offset} needs {encoded['input_ids'].size(1)} tokens, "
                    f"above model limit {model.config.max_position_embeddings}")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(**encoded).last_hidden_state

            for local_index, record in enumerate(batch_records):
                row_index = offset + local_index
                token_count = int(encoded["attention_mask"][local_index].sum())
                num_spans = (token_count + SPAN_TOKENS - 1) // SPAN_TOKENS
                if num_spans > MAX_SPANS:
                    raise RuntimeError(
                        f"hadm_id={record['hadm_id']} needs {num_spans} spans; "
                        f"increase NOTE_MAX_SPANS above {MAX_SPANS}; no truncation performed")
                token_lengths.append(token_count)
                valid_hidden = hidden[local_index, :token_count].float()
                for span_index, start in enumerate(range(0, token_count, SPAN_TOKENS)):
                    stop = min(start + SPAN_TOKENS, token_count)
                    memory[row_index, span_index] = valid_hidden[start:stop].mean(0).cpu().half()
                    mask[row_index, span_index] = True
                    counts[row_index, span_index] = stop - start

                reconstructed = (
                    memory[row_index].float()
                    * counts[row_index].float().unsqueeze(-1)
                ).sum(0) / counts[row_index].sum().float()
                baseline = torch.tensor(record["note_embedding"], dtype=torch.float32)
                cosine_values.append(float(F.cosine_similarity(reconstructed, baseline, dim=0)))

            print(
                f"[NOTE-MEMORY] rows={min(offset + BATCH_SIZE, len(records))}/{len(records)} "
                f"max_tokens={max(token_lengths)} min_cos={min(cosine_values):.6f}",
                flush=True)

    min_cosine = min(cosine_values)
    if min_cosine < MIN_MEAN_COSINE:
        raise RuntimeError(
            f"span reconstruction vs v22 mean embedding cosine {min_cosine:.6f} "
            f"below {MIN_MEAN_COSINE}; model/tokenizer provenance does not match")

    metadata = {
        "data_repo": DATA_REPO,
        "data_file": DATA_FILE,
        "source_sha256": sha256(source),
        "embed_model": MODEL_ID,
        "embed_revision": MODEL_REVISION,
        "span_tokens": str(SPAN_TOKENS),
        "max_spans": str(MAX_SPANS),
        "hidden_size": str(hidden_size),
        "records": str(len(records)),
        "max_tokens_observed": str(max(token_lengths)),
        "min_v22_mean_cosine": f"{min_cosine:.8f}",
        "mean_v22_mean_cosine": f"{sum(cosine_values) / len(cosine_values):.8f}",
    }
    save_file({
        "memory": memory.contiguous(),
        "mask": mask.contiguous(),
        "span_token_counts": counts.contiguous(),
        "hadm_ids": hadm_ids.contiguous(),
    }, OUTPUT_FILE, metadata=metadata)

    readme = Path("README.md")
    readme.write_text(
        "# Fawkes v23 Clinical ModernBERT note memory\n\n"
        "Private fixed-width span embeddings for the v23 mean-vs-attention ablation.\n\n"
        f"- Source: `{DATA_REPO}/{DATA_FILE}`\n"
        f"- Encoder: `{MODEL_ID}` at `{MODEL_REVISION}`\n"
        f"- Shape: `{tuple(memory.shape)}` (`float16`)\n"
        f"- Span width: `{SPAN_TOKENS}` contextual tokens\n"
        f"- Maximum observed note length: `{max(token_lengths)}` tokens\n"
        f"- Minimum cosine against the stored v22 mean: `{min_cosine:.8f}`\n"
        "- Notes were not truncated.\n",
        encoding="utf-8")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(OUTPUT_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=OUTPUT_FILE, path_in_repo=OUTPUT_FILE,
        repo_id=OUTPUT_REPO, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=str(readme), path_in_repo="README.md",
        repo_id=OUTPUT_REPO, repo_type="dataset")
    print(f"[DONE] https://huggingface.co/datasets/{OUTPUT_REPO}", flush=True)


if __name__ == "__main__":
    main()
