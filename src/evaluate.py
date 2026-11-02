"""Evaluate ONE chosen checkpoint on the test split, once, and report it to W&B.

Usage:
    python src/evaluate.py --checkpoint checkpoints/<run_name>/<file>.ckpt

What it does:
  1. prints the majority-class baseline on validation and test (majority class taken
     from the TRAIN labels, as a real baseline would be);
  2. loads the checkpoint (architecture + hyperparameters are stored inside it);
  3. computes validation metrics (sanity check: must match best_val_* of the run) and
     test_loss / test_acc;
  4. writes test_acc / test_loss into the summary of the ORIGINAL W&B training run
     (run id stored in the checkpoint) via the public API; no new run is created.

The test set is used here and nowhere else. To avoid looking at the test set repeatedly,
the script refuses to run if that W&B run already has a test_acc (override with --force).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import lightning as L
import torch
import wandb
from torch.utils.data import DataLoader

from data import ARTDataset, collate_mlp, collate_rnn, get_vocab_and_embeddings, label_to_index, load_art
from train import model_class


def majority_baseline(train_labels: list[int], eval_labels: list[int]) -> float:
    """Accuracy of always predicting the most frequent TRAIN label."""
    majority = Counter(train_labels).most_common(1)[0][0]
    return sum(label == majority for label in eval_labels) / len(eval_labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "nlp-hs26-project1"))
    p.add_argument("--no_wandb", action="store_true", help="only print, do not write to W&B")
    p.add_argument("--force", action="store_true", help="evaluate even if the run already has test_acc")
    args = p.parse_args()

    # Our own checkpoint file, so weights_only=False is safe; we only read the hyperparameters.
    hparams = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]
    run_id = hparams["wandb_run_id"]
    print(f"Checkpoint: {args.checkpoint}\nArchitecture: {hparams['arch']} | W&B run id: {run_id}")

    # ---- Look up the original W&B run first (and enforce "test only once") ----------
    api_run = None
    if not args.no_wandb:
        api = wandb.Api()
        entity = args.wandb_entity or api.default_entity
        api_run = api.run(f"{entity}/{args.wandb_project}/{run_id}")
        if "test_acc" in api_run.summary and not args.force:
            raise SystemExit(f"Run {api_run.name} already has test_acc={api_run.summary['test_acc']}. "
                             "The test set should be evaluated once; use --force to override.")

    # ---- Data ------------------------------------------------------------------------
    ds = load_art()
    # Same vocabulary/embeddings as in training (built from the train split, cached).
    itos, embeddings = get_vocab_and_embeddings(ds["train"], args.cache_dir)
    stoi = {w: i for i, w in enumerate(itos)}

    train_labels = [label_to_index(x) for x in ds["train"]["label"]]
    for split in ("validation", "test"):
        labels = [label_to_index(x) for x in ds[split]["label"]]
        print(f"Majority-class baseline ({split}): {majority_baseline(train_labels, labels):.4f}")

    collate = collate_mlp if hparams["arch"] == "mlp" else collate_rnn
    loaders = {split: DataLoader(ARTDataset(ds[split], stoi), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate)
               for split in ("validation", "test")}

    # ---- Model + evaluation ------------------------------------------------------------
    model = model_class(hparams["arch"]).load_from_checkpoint(
        args.checkpoint, embedding_matrix=embeddings, map_location="cpu")
    trainer = L.Trainer(accelerator=args.accelerator, devices=1, logger=False, enable_progress_bar=False)
    val = trainer.validate(model, loaders["validation"], verbose=False)[0]
    test = trainer.test(model, loaders["test"], verbose=False)[0]
    print(f"Validation: loss={val['val_loss']:.4f} acc={val['val_acc']:.4f}")
    print(f"Test:       loss={test['test_loss']:.4f} acc={test['test_acc']:.4f}")

    if api_run is not None:
        api_run.summary["test_acc"] = test["test_acc"]
        api_run.summary["test_loss"] = test["test_loss"]
        api_run.summary["test_checkpoint"] = args.checkpoint
        api_run.summary.update()  # persists the summary of the existing run
        print(f"Wrote test_acc/test_loss to W&B run '{api_run.name}' ({api_run.url})")


if __name__ == "__main__":
    main()
