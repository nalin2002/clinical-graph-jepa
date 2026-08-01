"""Pretrain with masked patch prediction only.

Merged from ``graph_jepa_v{5,6}/pretrain.py``. v5's copy was v6's minus the
three note arguments, so the merged script is v6's: the localized-note variant
is the default and ``--no-note-embeddings`` selects the no-note one. Which
variant was trained determines the checkpoint name — plan section 5.3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import Config
from ..encoders import build_encoder
from ..model import GraphJEPA
from .loop import (
    PRETRAIN_STAGE,
    add_data_args,
    add_runtime_args,
    apply_common_train_args,
    build_optimizer,
    build_train_loader,
    checkpoint_filename,
    init_wandb,
    save_checkpoint,
    train_epochs,
)


def pretrain(args) -> Path:
    cfg = Config()
    cfg.encoder = args.encoder
    cfg.train.pretrain_epochs = args.epochs
    cfg.train.finetune_epochs = 0
    cfg.train.epochs = args.epochs
    apply_common_train_args(args, cfg)
    cfg.model.num_patches = args.num_patches
    cfg.model.patch_pe_dim = args.patch_pe_dim
    cfg.model.conv = args.conv
    cfg.model.gnn_backend = args.gnn_backend
    cfg.model.use_note_embeddings = not args.no_note_embeddings
    cfg.model.note_embedding_dim = args.note_embedding_dim
    cfg.model.note_ground_by = args.note_ground_by

    if cfg.train.pretrain_epochs <= 0:
        raise ValueError("--epochs must be positive for pretraining")
    if cfg.model.use_note_embeddings and cfg.model.note_embedding_dim <= 0:
        raise ValueError("--note-embedding-dim must be positive when notes are enabled")

    encoder = build_encoder(args.encoder, mock_dim=args.mock_dim, cache_dir=args.encoder_cache)
    cfg.model.base_in_dim = encoder.dim
    cfg.model.in_dim = encoder.dim + (
        cfg.model.note_embedding_dim if cfg.model.use_note_embeddings else 0
    )

    torch.manual_seed(cfg.train.seed)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    device = torch.device(args.device)

    checkpoint_name = checkpoint_filename(cfg, pretrain=True)
    dataset, train_loader = build_train_loader(args, cfg, encoder)
    print(
        f"Loaded {len(dataset)} graphs for masked pretraining "
        f"(encoder={args.encoder}, base_in_dim={cfg.model.base_in_dim}, "
        f"in_dim={cfg.model.in_dim}, use_note={cfg.model.use_note_embeddings}, "
        f"note_dim={cfg.model.note_embedding_dim}, "
        f"ground_by={cfg.model.note_ground_by}, "
        f"patches={cfg.model.num_patches}, batch_size={cfg.train.batch_size}, "
        f"gnn_backend={cfg.model.gnn_backend}, conv={cfg.model.conv})"
    )
    wandb_run = init_wandb(
        args,
        cfg,
        len(dataset),
        script="clinical_jepa.train.pretrain",
        checkpoint_name=checkpoint_name,
    )

    model = GraphJEPA(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    train_epochs(
        model,
        optimizer,
        train_loader,
        cfg,
        stage_name=PRETRAIN_STAGE,
        epochs=cfg.train.pretrain_epochs,
        use_revision=False,
        device=device,
        generator=generator,
        wandb_run=wandb_run,
    )
    ckpt_path = save_checkpoint(
        model,
        cfg,
        args.out,
        checkpoint_name=checkpoint_name,
        config_name="config_pretrain.json",
    )
    if wandb_run:
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain masked Graph-JEPA representations")
    add_data_args(p)
    add_runtime_args(p)
    p.add_argument("--encoder", choices=["mock", "bge", "sapbert"], default="mock")
    p.add_argument("--mock-dim", type=int, default=256)
    p.add_argument("--encoder-cache", default=".cache/clinical_jepa/encoder")
    p.add_argument("--conv", choices=["gine", "gat"], default="gine")
    p.add_argument("--gnn-backend", choices=["pyg", "torch"], default="pyg")
    p.add_argument("--num-patches", type=int, default=8)
    p.add_argument("--patch-pe-dim", type=int, default=8)
    p.add_argument(
        "--no-note-embeddings",
        action="store_true",
        help="Disable localized note_embedding features (the no-note variant)",
    )
    p.add_argument("--note-embedding-dim", type=int, default=768)
    p.add_argument(
        "--note-ground-by",
        choices=["prov", "name", "all"],
        default="prov",
        help="Which entity nodes receive the graph-level note embedding",
    )
    return p


def main(argv=None) -> None:
    pretrain(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
