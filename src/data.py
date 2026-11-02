"""Data pipeline for the ART (abductive commonsense reasoning) project.

Everything that touches raw data lives here:
    1. loading the Hugging Face dataset,
    2. tokenization / preprocessing,
    3. vocabulary + frozen GloVe embedding matrix,
    4. a torch Dataset and the two collate functions (one per architecture).

Run `python src/data.py` to print the dataset report (schema, labels, lengths, OOV rates).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from nltk.tokenize import TreebankWordTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

DATASET_NAME = "amarfurt/art-nlp-hs26"
GLOVE_NAME = "glove-wiki-gigaword-300"
TEXT_FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")

# Special tokens. <pad> MUST be index 0 because nn.Embedding(padding_idx=0) and
# pad_sequence(padding_value=0) both rely on it.
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
PAD_IDX, UNK_IDX, SEP_IDX = 0, 1, 2
SPECIAL_TOKENS = [PAD, UNK, SEP]

# Fixed seed for the random <sep> vector. It is deliberately independent of the
# training seed so that every run (and evaluate.py) uses the identical embedding matrix.
EMBEDDING_SEED = 1234

_tokenizer = TreebankWordTokenizer()
# TreebankWordTokenizer expects ONE sentence: it only splits off the period at the very
# end of the string. A few fields contain several sentences ("... the bed. Then ..."),
# which would leave tokens like "bed." (OOV). Splitting on sentence-final punctuation
# followed by whitespace fixes this without needing nltk's punkt download.
_sentence_split = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_art() -> DatasetDict:
    """Load the course version of ART (splits: train / validation / test).

    Labels are 1 or 2 (index of the correct hypothesis, 1-based). The test split is only
    ever touched by evaluate.py.
    """
    return load_dataset(DATASET_NAME)


def label_to_index(label: int) -> int:
    """Map the dataset label (1 or 2) to the 0-based index of the correct hypothesis.

    This 0/1 index is directly the CrossEntropyLoss target for the (B, 2) logits.
    """
    if label not in (1, 2):
        raise ValueError(f"Unexpected label {label!r}; expected 1 or 2.")
    return label - 1


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def is_punctuation(token: str) -> bool:
    """True if the token contains no letter or digit (e.g. '.', ',', '``', '--', '...').

    Tokens such as "n't" or "'s" contain letters and are kept (GloVe has vectors for them).
    """
    return not any(ch.isalnum() for ch in token)


def tokenize(text: str) -> list[str]:
    """Lowercase, Treebank-tokenize and drop punctuation-only tokens.

    Decisions (see README):
      * lowercase, because GloVe (wiki-gigaword) is uncased;
      * TreebankWordTokenizer, because GloVe was trained on Treebank-style tokens
        (splits "didn't" -> "did", "n't" and "dog's" -> "dog", "'s");
      * stopwords are KEPT: words like "not"/"never" can decide which hypothesis is right;
      * punctuation-only tokens are removed (they carry little meaning here);
      * no stemming/lemmatization (GloVe has vectors for inflected forms),
        no HTML cleaning (the texts are clean), no truncation (sentences are short).
    """
    tokens: list[str] = []
    for sentence in _sentence_split.split(text.lower().strip()):
        tokens.extend(t for t in _tokenizer.tokenize(sentence) if not is_punctuation(t))
    return tokens


# --------------------------------------------------------------------------------------
# Vocabulary and embeddings
# --------------------------------------------------------------------------------------
def load_glove():
    """Load GloVe 300d through gensim's downloader.

    gensim caches the download (~380 MB) in ~/gensim-data (override with the env var
    GENSIM_DATA_DIR), so it is only downloaded once per machine.
    """
    import gensim.downloader as api  # imported lazily: slow import, only needed once

    return api.load(GLOVE_NAME)


def build_vocab(train_split: Iterable[dict], glove) -> list[str]:
    """Build the vocabulary (index -> token list) from the TRAIN split only.

    A word enters the vocabulary iff it appears in the training data AND GloVe has a
    vector for it. Words without a GloVe vector would get a vector we cannot learn
    (embeddings are frozen), so they are mapped to <unk> instead. Validation/test words
    are never added: that would leak information from held-out data into the model.

    Args:
        train_split: iterable of examples (dicts with the four text fields).
        glove: gensim KeyedVectors (anything with `.key_to_index`).

    Returns:
        List of tokens; position i is the token with index i. Special tokens come first.
    """
    counts: Counter[str] = Counter()
    for example in train_split:
        for field in TEXT_FIELDS:
            counts.update(tokenize(example[field]))
    # Sort for a deterministic vocabulary order (frequency desc, then alphabetical).
    words = sorted((w for w in counts if w in glove.key_to_index), key=lambda w: (-counts[w], w))
    return SPECIAL_TOKENS + words


def build_embedding_matrix(itos: Sequence[str], glove, seed: int = EMBEDDING_SEED) -> torch.Tensor:
    """Create the (vocab_size, 300) frozen embedding matrix.

    * real words: their GloVe vector;
    * <pad>: zeros (also kept at zero by padding_idx; ignored by masking/packing anyway);
    * <unk>: zeros. A zero vector adds nothing to a mean and is a neutral "no information"
      input for the RNN. (Alternative would be the mean GloVe vector; zero is simpler.)
    * <sep>: a fixed random vector, so the RNN can recognise segment boundaries. Its
      scale matches the standard deviation of the GloVe vectors so it is neither
      negligible nor dominant.
    """
    dim = glove.vector_size
    matrix = np.zeros((len(itos), dim), dtype=np.float32)
    for idx, word in enumerate(itos):
        if word not in SPECIAL_TOKENS:
            matrix[idx] = glove[word]
    rng = np.random.default_rng(seed)
    std = float(matrix[len(SPECIAL_TOKENS):].std()) if len(itos) > len(SPECIAL_TOKENS) else 1.0
    matrix[SEP_IDX] = rng.normal(0.0, std, size=dim)
    return torch.from_numpy(matrix)


def get_vocab_and_embeddings(train_split: HFDataset, cache_dir: str = "cache") -> tuple[list[str], torch.Tensor]:
    """Return (itos, embedding_matrix), building them once and caching them on disk.

    Loading all 400k GloVe vectors takes about a minute; the cached file only contains
    our vocabulary and is loaded instantly. Delete the cache file to rebuild.
    """
    path = os.path.join(cache_dir, "vocab_glove300.pt")
    if os.path.exists(path):
        cached = torch.load(path)
        return cached["itos"], cached["embeddings"]

    print(f"Building vocabulary from the train split and loading {GLOVE_NAME} ...")
    glove = load_glove()
    itos = build_vocab(train_split, glove)
    embeddings = build_embedding_matrix(itos, glove)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"itos": itos, "embeddings": embeddings}, path)
    print(f"Vocabulary size: {len(itos)} (incl. {len(SPECIAL_TOKENS)} special tokens). Cached to {path}")
    return itos, embeddings


def encode(tokens: Sequence[str], stoi: dict[str, int]) -> list[int]:
    """Map tokens to indices; unknown tokens -> <unk>."""
    return [stoi.get(t, UNK_IDX) for t in tokens]


def oov_rate(split: Iterable[dict], stoi: dict[str, int]) -> float:
    """Fraction of tokens (over all four text fields) that map to <unk>."""
    total = unknown = 0
    for example in split:
        for field in TEXT_FIELDS:
            tokens = tokenize(example[field])
            total += len(tokens)
            unknown += sum(t not in stoi for t in tokens)
    return unknown / max(total, 1)


# --------------------------------------------------------------------------------------
# torch Dataset and collate functions
# --------------------------------------------------------------------------------------
class ARTDataset(Dataset):
    """Pre-tokenized ART split. Each item is a dict of index lists plus the 0/1 label.

    Tokenization is done once in __init__ (the dataset is small), so __getitem__ and the
    collate functions only do cheap tensor work.
    """

    def __init__(self, split: Iterable[dict], stoi: dict[str, int]) -> None:
        self.items: list[dict] = []
        for example in split:
            item = {field: encode(tokenize(example[field]), stoi) for field in TEXT_FIELDS}
            item["label"] = label_to_index(example["label"])
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _pad(sequences: list[list[int]]) -> torch.Tensor:
    """Pad a list of index lists to the longest one -> LongTensor (B, L_max)."""
    tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]
    return pad_sequence(tensors, batch_first=True, padding_value=PAD_IDX)


def collate_mlp(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for the EmbeddingMLP: the four texts are padded separately.

    Padding is dynamic (to the longest text of this batch, not of the whole dataset).
    No lengths are needed: the model averages over positions where index != <pad>.

    Returns: o1, o2, h1, h2 of shape (B, L_field) and labels of shape (B,).
    """
    out = {
        "o1": _pad([b["observation_1"] for b in batch]),
        "o2": _pad([b["observation_2"] for b in batch]),
        "h1": _pad([b["hypothesis_1"] for b in batch]),
        "h2": _pad([b["hypothesis_2"] for b in batch]),
    }
    out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


def build_rnn_sequence(o1: list[int], h: list[int], o2: list[int]) -> list[int]:
    """`O1 <sep> H <sep> O2`: narrative order, so the RNN reads the story as it happened."""
    return o1 + [SEP_IDX] + h + [SEP_IDX] + o2


def collate_rnn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for the RNNClassifier: two candidate sequences per example.

    Order is [ex0+h1, ex0+h2, ex1+h1, ex1+h2, ...], so the model's (2B,) scores can be
    reshaped to (B, 2) with column j = hypothesis j.

    Returns:
        input_ids: (2B, L_max) padded indices,
        lengths:   (2B,) true lengths (needed by pack_padded_sequence),
        labels:    (B,) index of the correct hypothesis.
    """
    sequences: list[list[int]] = []
    for b in batch:
        for hyp in ("hypothesis_1", "hypothesis_2"):
            sequences.append(build_rnn_sequence(b["observation_1"], b[hyp], b["observation_2"]))
    return {
        "input_ids": _pad(sequences),
        "lengths": torch.tensor([len(s) for s in sequences], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# --------------------------------------------------------------------------------------
# Dataset report (step 1 of the project)
# --------------------------------------------------------------------------------------
def length_stats(split: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Token-length statistics (after preprocessing) per text field and for the RNN input."""
    lengths: dict[str, list[int]] = {f: [] for f in TEXT_FIELDS}
    lengths["rnn_sequence"] = []
    for example in split:
        toks = {f: tokenize(example[f]) for f in TEXT_FIELDS}
        for f in TEXT_FIELDS:
            lengths[f].append(len(toks[f]))
        for h in ("hypothesis_1", "hypothesis_2"):
            # +2 for the two <sep> tokens
            lengths["rnn_sequence"].append(len(toks["observation_1"]) + len(toks[h]) + len(toks["observation_2"]) + 2)
    return {
        name: {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": int(np.max(v)),
            "min": int(np.min(v)),
        }
        for name, v in lengths.items()
    }


def main() -> None:
    """Print the dataset report: schema, label distribution, length statistics, OOV rates."""
    ds = load_art()
    print(ds)
    print("Features:", ds["train"].features)
    for name, split in ds.items():
        counts = Counter(split["label"])
        dist = {k: f"{v} ({v / len(split):.1%})" for k, v in sorted(counts.items())}
        print(f"[{name}] label distribution: {dist}")

    itos, _ = get_vocab_and_embeddings(ds["train"])
    stoi = {w: i for i, w in enumerate(itos)}
    print(f"Vocabulary size: {len(itos)}")
    for name, split in ds.items():
        print(f"[{name}] OOV rate: {oov_rate(split, stoi):.2%}")
        for field, s in length_stats(split).items():
            print(f"[{name}] {field:15s} " + "  ".join(f"{k}={v:.1f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
