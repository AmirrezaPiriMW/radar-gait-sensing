"""Apply a trained ensemble to a different dataset (e.g. coupled models on decoupled data).

    python scripts/cross_predict.py --models results --data data/gait_dataset_decoupled.npz \
        --channels mono --out results/cross/coupled_mono_on_decoupled.npz

The output has the same keys as results/predictions_<channels>.npz. Target
standardisation comes from the ensemble's own training data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaitradar.model import GaitNet, ensemble_predict  # noqa: E402
from train import SPLITS, TARGETS, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="results directory of the trained ensemble")
    ap.add_argument("--data", required=True)
    ap.add_argument("--channels", choices=["mono", "multi"], required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = np.load(Path(a.models) / f"predictions_{a.channels}.npz")
    y_mean, y_std = src["y_mean"], src["y_std"]
    x, y, split, _ = load(a.data, a.channels)
    X = torch.from_numpy(x.astype(np.float32))
    models = []
    for k in range(a.members):
        m = GaitNet(in_ch=x.shape[1])
        m.load_state_dict(torch.load(Path(a.models) / "models" / f"{a.channels}_member{k}.pt", weights_only=True))
        models.append(m)
    pred = ensemble_predict(models, X)
    res = {"y_mean": y_mean, "y_std": y_std, "targets": np.array(TARGETS)}
    for s in SPLITS:
        msk = split == s
        res[f"{s}_y"] = y[msk]
        res[f"{s}_mu"] = pred["mu"][msk] * y_std + y_mean
        res[f"{s}_var"] = pred["var"][msk] * y_std ** 2
        res[f"{s}_aleatoric"] = pred["aleatoric"][msk] * y_std ** 2
        res[f"{s}_epistemic"] = pred["epistemic"][msk] * y_std ** 2
        res[f"{s}_index"] = np.flatnonzero(msk)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **res)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
