# NLP Project 1: Abductive commonsense reasoning (ART) with word embeddings and RNNs

Given the start of a story (observation **O1**) and its end (**O2**), choose which of two
hypotheses (**H1**, **H2**) better explains what happened in between. This is a binary choice.

Dataset: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)
with train 16,801 / validation 1,000 / test 1,532 examples. Fields: `observation_1`,
`observation_2`, `hypothesis_1`, `hypothesis_2`, `label` ∈ {1, 2}. Labels are balanced
(train 50.4% label 1, validation 50.8%, test 51.0%; see `python src/data.py`).

Two architectures, both on top of the **same frozen GloVe 300d embeddings**:

1. **EmbeddingMLP**: mean-pooled embeddings with a 2-layer ReLU classifier (only the classifier is trained).
2. **RNNClassifier**: a 2-layer LSTM (or GRU) followed by a 2-layer ReLU classifier (RNN and classifier are trained).

## Repository structure

```
README.md                  this file
EXPECTATIONS.md            my expected test accuracies (written by me)
requirements.txt
pytest.ini                 makes src/ importable in tests
configs/sweep_mlp.yaml     W&B grid sweep, architecture 1 (18 runs)
configs/sweep_rnn.yaml     W&B grid sweep, architecture 2 (12 runs)
src/data.py                loading, tokenization, vocab, embedding matrix, Dataset, collate fns
src/models.py              LightningModule base class + EmbeddingMLP + RNNClassifier
src/train.py               training entry point (standalone and W&B sweep agent)
src/evaluate.py            one-time test evaluation of a chosen checkpoint, writes to W&B
scripts/run_seeds.sh       re-run one config with seeds 42, 43, 44
scripts/create_sweep.py    create a W&B sweep that runs with the current (venv) Python
notebooks/01_data_exploration.ipynb   schema, label balance, lengths, OOV, examples
notebooks/02_results.ipynb            pulls runs from W&B: tables + val acc vs lr plot
tests/                     pytest unit tests (data pipeline + model shapes)
```

Generated at runtime (git-ignored): `cache/` (vocabulary + embedding matrix), `checkpoints/<run_name>/`,
`wandb/` (local W&B files).

## Setup

### Local / HSLU GPUHub (terminal)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
wandb login                          # paste your API key once
export WANDB_ENTITY=<your-entity>    # optional; defaults to your W&B default entity
export WANDB_PROJECT=nlp-hs26-project1
```

### Local Windows machine with an NVIDIA GPU (PowerShell)

```powershell
conda deactivate                         # if Anaconda's (base) is active
.\.venv\Scripts\Activate.ps1             # prompt must show (.venv)
pip install -r requirements.txt
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
wandb login
```

On Windows, create sweeps with `python scripts/create_sweep.py ...` instead of `wandb sweep ...`
(see step 4). `wandb agent` starts each run with a bare `python`, which on Windows resolves to the
global Python installation even inside an activated venv. The script writes the venv's full
interpreter path into the sweep.

The first run downloads the dataset (Hugging Face cache) and GloVe (~380 MB, cached by gensim in
`~/gensim-data`), then writes `cache/vocab_glove300.pt`. After that, every run starts in seconds.

### Google Colab

```python
!git clone https://github.com/CorsinAI/nlp-hs26-project1-art.git project
%cd project
!pip install -q -r requirements.txt      # Colab already has torch; pip keeps it
import wandb; wandb.login()
!python src/train.py --arch rnn --lr 1e-3 --hidden_dim 256 --dropout 0.3
```

Sweeps work the same way in Colab (`!wandb sweep ...`, `!wandb agent ...`).

### Tests

```bash
pytest -q
```

## Running everything

All commands are run from the repository root.

**1. Dataset report** (schema, label distribution, length statistics, OOV rates):

```bash
python src/data.py
```

**2. Smoke tests** (CPU is fine, no W&B account needed):

```bash
python src/train.py --arch mlp --fast_dev_run --wandb_mode offline
python src/train.py --arch rnn --fast_dev_run --wandb_mode offline
python src/train.py --arch mlp --subset 200 --max_epochs 3 --wandb_mode offline
python src/train.py --arch rnn --subset 200 --max_epochs 3 --wandb_mode offline
```

**3. Single training run:**

```bash
python src/train.py --arch mlp --lr 1e-3 --hidden_dim 256 --dropout 0.3
python src/train.py --arch rnn --lr 1e-3 --hidden_dim 256 --dropout 0.3
# options: --cell gru  --bidirectional  --input_format concat (mlp ablation)
#          --scheduler linear_warmup  --batch_size  --seed  --num_workers  --max_epochs
```

While a run is training on the GPU, check memory with `nvidia-smi` to confirm that batch size 256
fits (or try a larger batch, see design decisions).

**4. Hyperparameter sweeps** (W&B grid, metric `val_loss` minimized):

```bash
python scripts/create_sweep.py configs/sweep_mlp.yaml   # prints the agent command
wandb agent <entity>/nlp-hs26-project1/<sweep_id>
python scripts/create_sweep.py configs/sweep_rnn.yaml
wandb agent <entity>/nlp-hs26-project1/<sweep_id>
```

On Linux/Colab, `wandb sweep configs/sweep_mlp.yaml` works as well. Both YAMLs set
`project: nlp-hs26-project1`.

**5. Seed runs for the chosen configurations** (W&B group `final-<arch>`):

```bash
bash scripts/run_seeds.sh --arch mlp --lr <lr> --hidden_dim <h> --dropout <d>
bash scripts/run_seeds.sh --arch rnn --lr <lr> --hidden_dim <h> --dropout <d>
```

**6. Test evaluation, once per chosen checkpoint:**

```bash
python src/evaluate.py --checkpoint checkpoints/<run_name>/<run_id>-epoch=XX-val_loss=Y.ckpt
```

This prints the majority-class baseline (validation and test), the validation metrics of the
checkpoint as a sanity check, and the test loss and accuracy. It writes `test_acc` and `test_loss`
into the summary of the original training run. It refuses to evaluate a run that already has a
`test_acc` (use `--force` to override).

**7. Results:** open `notebooks/02_results.ipynb` to get the sweep tables, the best run per
architecture, seed mean/std and the validation-accuracy-vs-learning-rate plot.

## Design decisions

| Area | Decision | Justification |
|---|---|---|
| Embeddings | GloVe 300d (`glove-wiki-gigaword-300`) via gensim, frozen | Required by the task (only the classifier/RNN is trained). 300d is the richest GloVe variant. Freezing means words seen only at test time keep meaningful vectors relative to train words. |
| Tokenizer | nltk `TreebankWordTokenizer`, after a regex split into sentences | GloVe uses Treebank-style tokens ("did" + "n't", "dog" + "'s"). Treebank only splits the final period of a string, so multi-sentence fields are split first (no punkt download needed). |
| Case | Lowercase | GloVe wiki-gigaword is uncased. |
| Stopwords | Kept | Negations and function words ("not", "never", "before") often decide which hypothesis is right. |
| Punctuation | Punctuation-only tokens removed | They carry little meaning for plausibility. Tokens with letters ("n't", "'s") are kept. |
| Stemming / lemmatization | None | GloVe has vectors for inflected forms, and stemming would produce non-words without vectors. |
| HTML cleaning | None | The texts are clean crowd-sourced sentences. |
| Truncation | None | Max RNN input length is 58 tokens (p95 = 37), see `python src/data.py`. |
| Empty texts | Kept (6 train examples have placeholder hypotheses such as `{}.`) | Too few to matter. The code handles them: the mean pooling gives a zero vector, and the RNN input still contains 2 `<sep>` tokens. |
| Vocabulary | Train tokens that exist in GloVe, plus `<pad>`=0, `<unk>`=1, `<sep>`=2 | Using validation/test words would leak held-out data. A word without a GloVe vector cannot be learned (frozen embeddings), so it goes to `<unk>`. OOV: train 0.19%, validation 1.75%, test 1.78%, mostly proper names. |
| `<pad>` vector | Zeros, `padding_idx=0` | Excluded by masking (MLP) and packing (RNN) anyway. |
| `<unk>` vector | Zeros | Neutral "no information" input that adds nothing to a mean. Simpler than the mean GloVe vector. |
| `<sep>` vector | Fixed random vector (seed 1234, std of the GloVe vectors) | Lets the RNN detect segment boundaries. The fixed seed keeps it identical across runs and in evaluation. |
| Labels | 1/2 mapped to 0/1 | Index of the correct hypothesis, which is directly the CrossEntropy target. |
| Task formulation | Multiple choice with a shared scorer: one score per (O1, O2, H_i), giving (B, 2) logits, then CrossEntropy and argmax | The same function scores both hypotheses, so the model can't exploit hypothesis position and the output is symmetric. Softmax over the 2 scores models "which is more plausible". |
| Arch. 1 input | Mean of non-pad embeddings for O1, O2, H_i **separately**, concatenated (900d), then Linear, ReLU, Dropout, Linear(→1) | Keeps track of which text is which. One vector per text makes the model independent of length. |
| Arch. 1 ablation | `--input_format concat`: [O1; O2; H1; H2] (1200d), then MLP with 2 outputs | Checks whether the shared scorer helps. |
| Arch. 2 input | `O1 <sep> H_i <sep> O2` (narrative order), shape (2B, L, 300) | The RNN reads the story in the order the events happen. |
| Arch. 2 RNN | 2-layer unidirectional LSTM, `batch_first`, `pack_padded_sequence` with the true lengths; flags `--cell gru`, `--bidirectional` | Packing makes the final hidden state the state after the last real token, not after padding. |
| Arch. 2 output | Final hidden state of the top layer (both directions concatenated if bidirectional), then Linear, ReLU, Dropout, Linear(→1) | Standard sequence summary. `hidden_dim` sets both the RNN and the classifier width (one size hyperparameter). |
| Dropout | Only between the two classifier layers, never after the output layer | Dropout on the logits only adds noise. A single dropout hyperparameter keeps the sweep small. |
| Batching | Dynamic padding per batch in `collate_fn`, shuffle only train, default batch size 256 | Course rule: largest batch that fits in memory (verify on the GPU with `nvidia-smi`). Dynamic padding avoids wasting compute on padding. |
| num_workers | Default 0, configurable | The data is pre-tokenized in memory and collation is cheap. |
| Optimizer | AdamW, `weight_decay=0.01`, applied only to trainable parameters | Decoupled weight decay is a standard default. The frozen embeddings are excluded. |
| LR schedule | Constant by default; `--scheduler linear_warmup` (10% warmup, linear decay) | Keeps the comparison simple. Warmup is optional. |
| Metrics | `train_loss`/`train_acc` (step and epoch), `val_loss`/`val_acc` (epoch) with torchmetrics `Accuracy(task="multiclass")` | Micro accuracy = fraction correct. Note that `MulticlassAccuracy` defaults to *macro*. |
| Model selection | `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)`, `EarlyStopping(val_loss, patience=5)`, max 50 (MLP) / 30 (RNN) epochs | Validation loss is smoother than accuracy on 1,000 examples. The best checkpoint is re-validated and stored as `best_val_loss`/`best_val_acc`. |
| Checkpoints | Local `checkpoints/<run_name>/`, `log_model=False` | Saves W&B storage. The W&B run id is in the file name and the checkpoint's hyperparameters. |
| Seeds | `L.seed_everything(seed)`, final configs re-run with seeds 42, 43, 44 | Shows how much of a difference between configs is just noise. |
| Logging | W&B run names like `rnn-lstm_lr1e-3_h256_do0.3_s42`, group = architecture, tags, full config | Runs are easy to find and compare. |
| Sweeps | Grid: MLP lr {1e-2, 1e-3, 1e-4} × hidden {128, 256, 512} × dropout {0, 0.3}; RNN lr {1e-2, 1e-3, 1e-4} × hidden {128, 256} × dropout {0, 0.3} | Small enough to run in full, covers the most important hyperparameters. |
| Test set | Used only in `evaluate.py`, once per chosen checkpoint | Avoids selecting models on test data. The script refuses to re-evaluate a run. |
| Baseline | Majority class from the train labels, reported on validation and test | Reference point for a balanced binary task. |

## Reproduction

```bash
pip install -r requirements.txt && wandb login
pytest -q
python src/data.py
python scripts/create_sweep.py configs/sweep_mlp.yaml && wandb agent <entity>/nlp-hs26-project1/<id>
python scripts/create_sweep.py configs/sweep_rnn.yaml && wandb agent <entity>/nlp-hs26-project1/<id>
# choose the best config per architecture by best_val_loss (notebooks/02_results.ipynb)
bash scripts/run_seeds.sh --arch mlp --lr <lr> --hidden_dim <h> --dropout <d>
bash scripts/run_seeds.sh --arch rnn --lr <lr> --hidden_dim <h> --dropout <d>
python src/evaluate.py --checkpoint <chosen checkpoint>
```

## Tools used

**Claude Code** (Anthropic's AI coding assistant) was used to help write the code, tests and
documentation of this project. I reviewed every file and can explain every design decision.
