#!/usr/bin/env bash
# Re-run one configuration with seeds 42, 43, 44 (W&B group "final-<arch label>", tag "final").
# Seed variance shows how much of a difference between configs is just noise.
#
# Usage (from the repository root), pass the chosen hyperparameters:
#   bash scripts/run_seeds.sh --arch mlp --lr 1e-3 --hidden_dim 256 --dropout 0.3
#   bash scripts/run_seeds.sh --arch rnn --lr 1e-3 --hidden_dim 256 --dropout 0.3
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/run_seeds.sh --arch {mlp,rnn} --lr LR --hidden_dim H --dropout D [more train.py args]" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
for seed in 42 43 44; do
  echo "=== seed ${seed} ==="
  "$PYTHON" src/train.py "$@" --seed "${seed}" --group_prefix final- --tags final
done
