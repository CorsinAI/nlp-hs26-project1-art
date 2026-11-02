"""Models for ART: a shared LightningModule base class and the two architectures.

Multiple-choice formulation (used by both architectures by default):
    For each example we build two candidates (O1, O2, H1) and (O1, O2, H2). The SAME
    network ("scorer") maps each candidate to one scalar plausibility score. The two
    scores form (B, 2) logits, trained with CrossEntropyLoss against the index of the
    correct hypothesis; the prediction is the argmax.
    Why: the scorer cannot exploit the position of a hypothesis (h1 vs h2), it learns
    one notion of "plausible explanation", and softmax over the two scores directly
    models "which of the two is more plausible".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import lightning as L
from torch.nn.utils.rnn import pack_padded_sequence
from torchmetrics import Accuracy

from data import PAD_IDX


class ARTClassifier(L.LightningModule):
    """Shared training logic. Subclasses only implement `forward(batch) -> (B, 2) logits`.

    Args:
        embedding_matrix: (vocab_size, 300) GloVe matrix; wrapped in a FROZEN nn.Embedding.
        lr: AdamW learning rate.
        weight_decay: AdamW (decoupled) weight decay.
        scheduler: "constant" or "linear_warmup" (10% linear warmup, then linear decay to 0).
        **kwargs: any extra hyperparameters (e.g. seed, wandb_run_id); they are stored in
            the checkpoint via save_hyperparameters so evaluate.py can read them back.
    """

    def __init__(self, embedding_matrix: torch.Tensor, lr: float = 1e-3, weight_decay: float = 0.01,
                 scheduler: str = "constant", **kwargs) -> None:
        super().__init__()
        # The embedding matrix is not a hyperparameter (and is large): don't store it in
        # hparams. Its values are still saved in the checkpoint's state_dict.
        self.save_hyperparameters(ignore=["embedding_matrix"])
        # freeze=True: requires_grad=False, so the GloVe vectors are never updated.
        # padding_idx=0: the <pad> row stays zero and gets no gradient.
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=PAD_IDX)
        self.loss_fn = nn.CrossEntropyLoss()
        # Separate metric objects per stage so their running states don't mix.
        # task="multiclass" averages "micro" = plain fraction of correct predictions.
        # (Careful: torchmetrics' MulticlassAccuracy class defaults to MACRO averaging.)
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch)                       # (B, 2)
        loss = self.loss_fn(logits, batch["labels"])
        preds = logits.argmax(dim=1)               # index of the predicted hypothesis
        return loss, preds, batch["labels"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        bs = labels.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        bs = labels.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        bs = labels.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        # Only trainable parameters: the frozen embedding is excluded, so weight decay is
        # never applied to (and can never change) the GloVe vectors.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "constant":
            return optimizer
        if self.hparams.scheduler == "linear_warmup":
            # Planned over max_epochs (early stopping may end training before the decay finishes).
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = max(1, int(0.1 * total_steps))

            def lr_factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        raise ValueError(f"Unknown scheduler {self.hparams.scheduler!r}")


def masked_mean(embedded: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Mean over the non-<pad> positions.

    embedded: (B, L, D), ids: (B, L) -> (B, D). <unk> tokens have a zero vector but still
    count in the denominator. Empty texts (length 0) give a zero vector (clamp avoids 0/0).
    """
    mask = (ids != PAD_IDX).unsqueeze(-1).float()        # (B, L, 1)
    summed = (embedded * mask).sum(dim=1)                 # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)               # (B, 1)
    return summed / counts


class EmbeddingMLP(ARTClassifier):
    """Architecture 1: averaged frozen GloVe embeddings + 2-layer MLP (only the MLP is trained).

    input_format="shared" (default, multiple-choice):
        candidate i = [mean(O1); mean(O2); mean(H_i)]  (900 dims)
        score_i = Linear(900, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)
        logits = [score_1, score_2]
        O1, O2 and H are averaged SEPARATELY so the model knows which text is which
        (averaging everything together would lose that information).
    input_format="concat" (ablation):
        [mean(O1); mean(O2); mean(H1); mean(H2)] (1200 dims) -> 2-layer MLP -> 2 logits.
        Not symmetric in h1/h2: it must learn the same thing for both positions.
    """

    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 256, dropout: float = 0.3,
                 input_format: str = "shared", **kwargs) -> None:
        super().__init__(embedding_matrix, hidden_dim=hidden_dim, dropout=dropout,
                         input_format=input_format, **kwargs)
        if input_format not in ("shared", "concat"):
            raise ValueError(f"Unknown input_format {input_format!r}")
        dim = embedding_matrix.size(1)
        n_texts, n_outputs = (3, 1) if input_format == "shared" else (4, 2)
        # No dropout after the output layer: it would just add noise to the logits.
        self.classifier = nn.Sequential(
            nn.Linear(n_texts * dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        o1, o2, h1, h2 = (masked_mean(self.embedding(batch[k]), batch[k]) for k in ("o1", "o2", "h1", "h2"))
        if self.hparams.input_format == "concat":
            return self.classifier(torch.cat([o1, o2, h1, h2], dim=-1))           # (B, 2)
        candidates = torch.stack([torch.cat([o1, o2, h1], dim=-1),
                                  torch.cat([o1, o2, h2], dim=-1)], dim=1)       # (B, 2, 900)
        return self.classifier(candidates).squeeze(-1)                             # (B, 2)


class RNNClassifier(ARTClassifier):
    """Architecture 2: frozen GloVe -> 2-layer LSTM/GRU -> 2-layer MLP scorer.

    Each candidate is the sequence `O1 <sep> H_i <sep> O2` (narrative order). The RNN reads
    it, and the final hidden state of the top layer (both directions concatenated if
    bidirectional) is scored by Linear -> ReLU -> Dropout -> Linear(hidden, 1).
    hidden_dim is used both as the RNN hidden size and as the classifier hidden size, so
    the sweep has a single size hyperparameter.
    """

    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 256, dropout: float = 0.3,
                 cell: str = "lstm", bidirectional: bool = False, num_layers: int = 2, **kwargs) -> None:
        super().__init__(embedding_matrix, hidden_dim=hidden_dim, dropout=dropout, cell=cell,
                         bidirectional=bidirectional, num_layers=num_layers, **kwargs)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        # Dropout is applied only in the classifier (one dropout hyperparameter to tune).
        self.rnn = rnn_cls(input_size=embedding_matrix.size(1), hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(num_directions * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, lengths = batch["input_ids"], batch["lengths"]      # (2B, L), (2B,)
        embedded = self.embedding(input_ids)                            # (2B, L, 300)
        # Packing makes the RNN stop at each sequence's true length, so the final hidden
        # state is the state after the last REAL token, not after trailing padding.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):   # LSTM returns (h_n, c_n); GRU returns h_n
            hidden = hidden[0]
        # h_n: (num_layers * num_directions, 2B, H). The last 1 (or 2) rows are the top layer.
        if self.hparams.bidirectional:
            final = torch.cat([hidden[-2], hidden[-1]], dim=-1)          # forward + backward
        else:
            final = hidden[-1]
        scores = self.classifier(final).squeeze(-1)                     # (2B,)
        return scores.view(-1, 2)                                       # (B, 2): column j = hypothesis j
