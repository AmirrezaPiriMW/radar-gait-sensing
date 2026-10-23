#!/usr/bin/env bash
# Reproduce every result and figure (about 4 h on a 2-core laptop CPU).
set -euo pipefail

# 1) Main experiment: speed-coupled (Boulic) gait
python scripts/generate_dataset.py --out data/gait_dataset.npz
python scripts/train.py --channels mono
python scripts/train.py --channels multi
python scripts/evaluate.py
python scripts/plot_signatures.py

# 2) Decoupling experiment: speed and cadence drawn independently
python scripts/generate_dataset.py --decoupled --out data/gait_dataset_decoupled.npz
python scripts/train.py --data data/gait_dataset_decoupled.npz --channels mono  --out results/decoupled
python scripts/train.py --data data/gait_dataset_decoupled.npz --channels multi --out results/decoupled
for ch in mono multi; do
  python scripts/cross_predict.py --models results --data data/gait_dataset_decoupled.npz \
      --channels $ch --out results/cross/coupled_${ch}_on_decoupled.npz
  python scripts/cross_predict.py --models results/decoupled --data data/gait_dataset.npz \
      --channels $ch --out results/cross/decoupled_${ch}_on_coupled.npz
done
python scripts/evaluate_decoupled.py

python -m pytest -q tests
