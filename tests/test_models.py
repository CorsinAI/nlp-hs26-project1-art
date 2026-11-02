"""Forward-pass tests for both architectures (random data, tiny vocabulary)."""

import pytest
import torch

from data import PAD_IDX, SEP_IDX
from models import EmbeddingMLP, RNNClassifier

VOCAB, DIM, B = 50, 300, 4


@pytest.fixture
def emb() -> torch.Tensor:
    m = torch.randn(VOCAB, DIM)
    m[PAD_IDX] = 0
    return m


def mlp_batch() -> dict[str, torch.Tensor]:
    batch = {k: torch.randint(3, VOCAB, (B, L)) for k, L in (("o1", 7), ("o2", 9), ("h1", 5), ("h2", 6))}
    batch["h2"][0, 3:] = PAD_IDX  # some padding
    batch["labels"] = torch.tensor([0, 1, 1, 0])
    return batch


def rnn_batch() -> dict[str, torch.Tensor]:
    lengths = torch.tensor([12, 10, 8, 12, 5, 7, 11, 9])  # 2B candidates
    ids = torch.randint(3, VOCAB, (2 * B, 12))
    for i, n in enumerate(lengths):
        ids[i, n:] = PAD_IDX
    ids[:, 2] = SEP_IDX
    return {"input_ids": ids, "lengths": lengths, "labels": torch.tensor([0, 1, 1, 0])}


@pytest.mark.parametrize("input_format", ["shared", "concat"])
def test_mlp_forward_shape(emb, input_format):
    model = EmbeddingMLP(emb, hidden_dim=16, dropout=0.3, input_format=input_format)
    logits = model(mlp_batch())
    assert logits.shape == (B, 2)


@pytest.mark.parametrize("cell,bidirectional", [("lstm", False), ("gru", False), ("lstm", True)])
def test_rnn_forward_shape(emb, cell, bidirectional):
    model = RNNClassifier(emb, hidden_dim=16, dropout=0.3, cell=cell, bidirectional=bidirectional)
    logits = model(rnn_batch())
    assert logits.shape == (B, 2)


def test_embeddings_frozen_and_excluded_from_optimizer(emb):
    model = RNNClassifier(emb, hidden_dim=16)
    assert not model.embedding.weight.requires_grad
    optimizer = model.configure_optimizers()
    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(model.embedding.weight) not in optimized
    assert id(next(model.rnn.parameters())) in optimized


def test_loss_backward_updates_only_trainable_params(emb):
    model = EmbeddingMLP(emb, hidden_dim=16)
    batch = mlp_batch()
    loss = model.loss_fn(model(batch), batch["labels"])
    loss.backward()
    assert model.embedding.weight.grad is None
    assert model.classifier[0].weight.grad is not None


def test_shared_scorer_is_symmetric(emb):
    """Swapping h1 and h2 must swap the two logits (same scorer for both candidates)."""
    model = EmbeddingMLP(emb, hidden_dim=16).eval()
    batch = mlp_batch()
    swapped = {**batch, "h1": batch["h2"], "h2": batch["h1"]}
    assert torch.allclose(model(batch), model(swapped).flip(dims=[1]), atol=1e-6)


def test_rnn_ignores_padding(emb):
    """Extra trailing padding must not change the output (thanks to packing)."""
    model = RNNClassifier(emb, hidden_dim=16).eval()
    batch = rnn_batch()
    padded = {**batch, "input_ids": torch.cat([batch["input_ids"], torch.zeros(2 * B, 5, dtype=torch.long)], dim=1)}
    assert torch.allclose(model(batch), model(padded), atol=1e-6)


def test_hparams_saved(emb):
    model = RNNClassifier(emb, hidden_dim=16, lr=3e-4, cell="gru", seed=7)
    for key, value in {"hidden_dim": 16, "lr": 3e-4, "cell": "gru", "seed": 7}.items():
        assert model.hparams[key] == value
    assert "embedding_matrix" not in model.hparams


def test_accuracy_is_plain_fraction_correct(emb):
    """Micro accuracy: 3 of 4 correct = 0.75 (macro averaging would give 0.5 here)."""
    model = EmbeddingMLP(emb, hidden_dim=16)
    acc = model.val_acc(torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 1]))
    assert acc.item() == pytest.approx(0.75)
