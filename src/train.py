"""Train one model (EmbeddingMLP or RNNClassifier) on ART.

Works in two modes:
  * standalone:    python src/train.py --arch rnn --lr 1e-3 --hidden_dim 256 --dropout 0.3
  * W&B sweep:     wandb agent <entity>/<project>/<sweep_id>
    The agent starts this script with the sweep values as CLI args (e.g. --lr=0.001) and
    sets WANDB_SWEEP_ID. We additionally copy wandb.config back into `args`, so the values
    the sweep chose are guaranteed to be the ones used.

The test split is NEVER loaded into a DataLoader here; model selection uses val_loss only.
"""

from __future__ import annotations

import argparse
import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from data import ARTDataset, collate_mlp, collate_rnn, get_vocab_and_embeddings, load_art, oov_rate
from models import ARTClassifier, EmbeddingMLP, RNNClassifier

DEFAULT_MAX_EPOCHS = {"mlp": 50, "rnn": 30}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model
    p.add_argument("--arch", choices=["mlp", "rnn"], required=True)
    p.add_argument("--input_format", choices=["shared", "concat"], default="shared",
                   help="mlp only: shared scorer over 2 candidates (default) or concat ablation")
    p.add_argument("--cell", choices=["lstm", "gru"], default="lstm", help="rnn only")
    p.add_argument("--bidirectional", action="store_true", help="rnn only")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    # Optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--scheduler", choices=["constant", "linear_warmup"], default="constant")
    p.add_argument("--batch_size", type=int, default=256,
                   help="course rule: largest batch that fits in memory (verify on the GPU)")
    p.add_argument("--max_epochs", type=int, default=None, help="default: 50 (mlp) / 30 (rnn)")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience on val_loss")
    p.add_argument("--seed", type=int, default=42)
    # Data / hardware
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 is enough: data is pre-tokenized in memory, collation is cheap")
    p.add_argument("--subset", type=int, default=None, help="use only the first N train/val examples (smoke tests)")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--fast_dev_run", action="store_true", help="1 train + 1 val batch, no checkpoints")
    # Weights & Biases
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "nlp-hs26-project1"))
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None,
                   help="default: W&B decides (env var WANDB_MODE, else online)")
    p.add_argument("--group_prefix", default="", help='e.g. "final-" -> W&B group "final-rnn-lstm"')
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args(argv)
    if args.max_epochs is None:
        args.max_epochs = DEFAULT_MAX_EPOCHS[args.arch]
    return args


def arch_label(args: argparse.Namespace) -> str:
    """Short architecture name, used as W&B group, e.g. 'mlp-shared', 'rnn-lstm', 'rnn-gru-bi'."""
    if args.arch == "mlp":
        return f"mlp-{args.input_format}"
    return f"rnn-{args.cell}" + ("-bi" if args.bidirectional else "")


def format_lr(lr: float) -> str:
    """0.001 -> '1e-3' (compact, for run names)."""
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_run_name(args: argparse.Namespace) -> str:
    """Descriptive run name, e.g. 'rnn-lstm_lr1e-3_h256_do0.3_s42'."""
    name = f"{arch_label(args)}_lr{format_lr(args.lr)}_h{args.hidden_dim}_do{args.dropout:g}_s{args.seed}"  # :g -> 0 and 0.0 both give "do0"
    if args.scheduler != "constant":
        name += f"_{args.scheduler}"
    return name


def make_dataloaders(args: argparse.Namespace, train_split, val_split, stoi: dict[str, int]) -> tuple[DataLoader, DataLoader]:
    """Train loader is shuffled (new order every epoch); validation is not (order is irrelevant)."""
    collate = collate_mlp if args.arch == "mlp" else collate_rnn
    train_loader = DataLoader(ARTDataset(train_split, stoi), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(ARTDataset(val_split, stoi), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)
    return train_loader, val_loader


def model_class(arch: str) -> type[ARTClassifier]:
    return {"mlp": EmbeddingMLP, "rnn": RNNClassifier}[arch]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    in_sweep = "WANDB_SWEEP_ID" in os.environ

    # ---- W&B run -------------------------------------------------------------------
    # We create the run ourselves (instead of letting WandbLogger do it) so we can read
    # the sweep's wandb.config before building the model. Under `wandb agent`, this
    # wandb.init() attaches to the run the agent prepared.
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=make_run_name(args),
        group=args.group_prefix + arch_label(args),
        tags=[args.arch, arch_label(args), "sweep" if in_sweep else "manual", *args.tags],
        config=vars(args),
        mode=args.wandb_mode,
    )
    if in_sweep:
        for key in vars(args):
            if key in run.config:
                setattr(args, key, run.config[key])
        run.name = make_run_name(args)
    run_name = make_run_name(args)
    # The summary value of val_loss = its minimum over epochs = the val_loss of the
    # checkpoint we keep. This is what the sweep's "metric: val_loss" then reports.
    run.define_metric("val_loss", summary="min")
    print(f"Run: {run_name} (id {run.id})\n{vars(args)}")

    L.seed_everything(args.seed, workers=True)

    # ---- Data ----------------------------------------------------------------------
    ds = load_art()
    train_split, val_split = ds["train"], ds["validation"]  # test split deliberately unused
    # Vocabulary is always built from the FULL train split (also with --subset), so that
    # smoke tests use the same embedding matrix as real runs.
    itos, embeddings = get_vocab_and_embeddings(train_split, args.cache_dir)
    stoi = {w: i for i, w in enumerate(itos)}
    if args.subset:
        train_split = train_split.select(range(min(args.subset, len(train_split))))
        val_split = val_split.select(range(min(args.subset, len(val_split))))
    oov = {"oov_train": oov_rate(train_split, stoi), "oov_val": oov_rate(val_split, stoi)}
    print(f"Vocab size {len(itos)} | OOV train {oov['oov_train']:.2%} | OOV val {oov['oov_val']:.2%}")
    train_loader, val_loader = make_dataloaders(args, train_split, val_split, stoi)

    # ---- Model ---------------------------------------------------------------------
    cls = model_class(args.arch)
    # All args are passed on and stored as hyperparameters in the checkpoint (evaluate.py
    # reads arch, batch_size and the W&B run id from there). Unused ones end up in **kwargs.
    model = cls(embeddings, **vars(args), wandb_run_id=run.id)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.config.update({"run_name": run_name, "vocab_size": len(itos), "n_train": len(train_split),
                       "n_val": len(val_split), "trainable_params": n_trainable, **oov},
                      allow_val_change=True)
    print(f"Trainable parameters: {n_trainable:,}")

    # ---- Trainer -------------------------------------------------------------------
    # log_model=False: checkpoints stay local (checkpoints/<run_name>/), nothing uploaded to W&B.
    logger = WandbLogger(experiment=run, log_model=False)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, run_name),
        filename=run.id + "-{epoch:02d}-{val_loss:.4f}",  # run id keeps re-runs of the same config apart
        monitor="val_loss", mode="min", save_top_k=1,
    )
    callbacks = [checkpoint_cb, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
                 LearningRateMonitor(logging_interval="step")]
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, train_loader, val_loader)

    # ---- Re-validate the best checkpoint -------------------------------------------
    # The last epoch is usually NOT the best one (early stopping waits `patience` epochs),
    # so we record the validation metrics of the checkpoint we actually keep.
    best_path = checkpoint_cb.best_model_path
    if best_path:  # empty with --fast_dev_run (no checkpointing)
        best_model = cls.load_from_checkpoint(best_path, embedding_matrix=embeddings, map_location="cpu")
        eval_trainer = L.Trainer(accelerator=args.accelerator, devices=1, logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False)
        best = eval_trainer.validate(best_model, val_loader, verbose=False)[0]
        run.summary.update({"best_val_loss": best["val_loss"], "best_val_acc": best["val_acc"],
                            "best_epoch": int(torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]),
                            "best_checkpoint": best_path})
        print(f"Best checkpoint: {best_path}\n  val_loss={best['val_loss']:.4f} val_acc={best['val_acc']:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
