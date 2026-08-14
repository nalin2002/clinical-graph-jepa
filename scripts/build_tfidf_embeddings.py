#!/usr/bin/env python3
"""Create a leakage-safe 768-d TF-IDF replacement for discharge-note vectors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


def split_indices(n: int, seed: int, test_frac: float, val_frac: float):
    order = np.random.RandomState(seed).permutation(n)
    n_test, n_val = int(test_frac * n), int(val_frac * n)
    return order[n_test + n_val :], order[n_test : n_test + n_val], order[:n_test]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    records = [json.loads(line) for line in args.input.open() if line.strip()]
    notes = [str(record.get("note") or "") for record in records]
    train_idx, val_idx, test_idx = split_indices(len(records), args.seed, args.test_frac, args.val_frac)
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.98, max_features=200_000,
    )
    x_train = vectorizer.fit_transform([notes[i] for i in train_idx])
    svd_dim = max(1, min(x_train.shape[0] - 1, x_train.shape[1], args.dim))
    svd = TruncatedSVD(n_components=svd_dim, n_iter=7, random_state=args.seed)
    normalizer = Normalizer(copy=False)
    normalizer.fit_transform(svd.fit_transform(x_train))
    z_all = normalizer.transform(svd.transform(vectorizer.transform(notes)))
    embeddings = np.zeros((len(records), args.dim), dtype=np.float32)
    embeddings[:, :svd_dim] = z_all.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record, vector in zip(records, embeddings):
            output = copy.deepcopy(record)
            output.update({
                "note_embedding": vector.tolist(),
                "embed_model": "tfidf_word_bigram_svd",
                "embed_dim": args.dim,
                "note_embedding_pooling": "tfidf_train_only_svd_l2",
            })
            stream.write(json.dumps(output, separators=(",", ":")) + "\n")
    args.metadata.write_text(json.dumps({
        "encoder": "tfidf", "vectorizer": "TfidfVectorizer",
        "ngram_range": [1, 2], "sublinear_tf": True, "min_df": 2,
        "max_df": 0.98, "max_features": 200000,
        "projection": "TruncatedSVD + L2 normalization",
        "fit_graphs": len(train_idx), "validation_graphs": len(val_idx),
        "test_graphs": len(test_idx), "split_seed": args.seed, "dim": args.dim,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "train": len(train_idx),
                      "val": len(val_idx), "test": len(test_idx),
                      "vocab": len(vectorizer.vocabulary_), "svd_dim": svd_dim}, indent=2))


if __name__ == "__main__":
    main()
