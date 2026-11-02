"""Tests for src/data.py. They use a tiny fake GloVe, so they run offline in seconds."""

import numpy as np
import torch
from gensim.models import KeyedVectors

from data import (
    PAD_IDX,
    SEP_IDX,
    SPECIAL_TOKENS,
    UNK_IDX,
    ARTDataset,
    build_embedding_matrix,
    build_rnn_sequence,
    build_vocab,
    collate_mlp,
    collate_rnn,
    label_to_index,
    tokenize,
)

TRAIN = [
    {"observation_1": "The dog was hungry.", "observation_2": "The dog slept well.",
     "hypothesis_1": "The dog didn't eat.", "hypothesis_2": "The dog ate dinner!", "label": 2},
    {"observation_1": "I went out.", "observation_2": "I got wet.",
     "hypothesis_1": "It rained.", "hypothesis_2": "It was sunny, never rainy.", "label": 1},
]
# "zebra" exists in the fake GloVe but appears only in the test split.
TEST = [
    {"observation_1": "A zebra ran.", "observation_2": "The zebra won.",
     "hypothesis_1": "The zebra trained.", "hypothesis_2": "The zebra slept.", "label": 1},
]


def fake_glove() -> KeyedVectors:
    """Fake 4-dim GloVe that knows most train words plus 'zebra' (but not 'dinner')."""
    words = ["the", "dog", "was", "hungry", "slept", "well", "did", "n't", "eat", "ate",
             "i", "went", "out", "got", "wet", "it", "rained", "sunny", "never", "zebra"]
    kv = KeyedVectors(vector_size=4)
    rng = np.random.default_rng(0)
    kv.add_vectors(words, rng.normal(size=(len(words), 4)).astype(np.float32))
    return kv


def make_vocab():
    itos = build_vocab(TRAIN, fake_glove())
    return itos, {w: i for i, w in enumerate(itos)}


def test_tokenize_lowercases_keeps_negation_drops_punctuation():
    assert tokenize("The dog DIDN'T eat, \"never\"!") == ["the", "dog", "did", "n't", "eat", "never"]


def test_tokenize_splits_multiple_sentences():
    # Without the sentence split, Treebank would leave "bed." as one token.
    assert tokenize("He went to bed. Then he slept.") == ["he", "went", "to", "bed", "then", "he", "slept"]


def test_label_mapping():
    assert label_to_index(1) == 0
    assert label_to_index(2) == 1


def test_special_token_indices():
    itos, _ = make_vocab()
    assert itos[:3] == SPECIAL_TOKENS
    assert (PAD_IDX, UNK_IDX, SEP_IDX) == (0, 1, 2)


def test_vocab_uses_only_train_words_that_are_in_glove():
    itos, stoi = make_vocab()
    assert "dog" in stoi
    assert "dinner" not in stoi  # in train, but not in GloVe -> <unk>
    assert "zebra" not in stoi   # in GloVe, but only in TEST -> must not leak into the vocab


def test_embedding_matrix_special_rows():
    glove = fake_glove()
    itos = build_vocab(TRAIN, glove)
    emb = build_embedding_matrix(itos, glove)
    assert emb.shape == (len(itos), 4)
    assert torch.all(emb[PAD_IDX] == 0)
    assert torch.all(emb[UNK_IDX] == 0)
    assert emb[SEP_IDX].abs().sum() > 0
    dog = itos.index("dog")
    assert torch.allclose(emb[dog], torch.tensor(glove["dog"]))
    # The <sep> vector is seeded -> identical when rebuilt.
    assert torch.equal(emb, build_embedding_matrix(itos, glove))


def test_dataset_unknown_words_map_to_unk():
    _, stoi = make_vocab()
    ds = ARTDataset(TEST, stoi)
    assert ds[0]["observation_1"][1] == UNK_IDX  # "zebra"
    assert ds[0]["label"] == 0


def test_collate_mlp_shapes_and_padding():
    _, stoi = make_vocab()
    batch = collate_mlp([ARTDataset(TRAIN, stoi)[i] for i in range(2)])
    assert batch["labels"].tolist() == [1, 0]
    # hypothesis_2 lengths: "the dog ate dinner" = 4, "it was sunny never rainy" = 5
    assert batch["h2"].shape == (2, 5)
    assert batch["h2"][0, 4].item() == PAD_IDX
    assert batch["h2"][1, 4].item() != PAD_IDX


def test_collate_rnn_shapes_order_and_lengths():
    _, stoi = make_vocab()
    items = [ARTDataset(TRAIN, stoi)[i] for i in range(2)]
    batch = collate_rnn(items)
    assert batch["input_ids"].shape[0] == 4  # 2 examples x 2 candidates
    assert batch["labels"].shape == (2,)
    # Row 1 is example 0 with hypothesis 2, in the order O1 <sep> H <sep> O2.
    expected = build_rnn_sequence(items[0]["observation_1"], items[0]["hypothesis_2"], items[0]["observation_2"])
    assert batch["lengths"][1].item() == len(expected)
    assert batch["input_ids"][1, : len(expected)].tolist() == expected
    assert (batch["input_ids"][1, len(expected):] == PAD_IDX).all()
    assert expected.count(SEP_IDX) == 2
    assert batch["input_ids"].shape[1] == batch["lengths"].max().item()  # dynamic padding


def test_empty_text_is_handled():
    # A few train hypotheses are just "{}." -> zero tokens after preprocessing.
    _, stoi = make_vocab()
    item = ARTDataset([{**TRAIN[0], "hypothesis_1": "{}."}], stoi)[0]
    assert item["hypothesis_1"] == []
    batch = collate_rnn([item])
    assert batch["lengths"].min().item() >= 2  # the two <sep> tokens are always there
