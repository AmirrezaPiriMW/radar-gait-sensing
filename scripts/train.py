"""Train a deep ensemble of heteroscedastic CNNs for speed and cadence.

    python scripts/train.py --channels mono    # monostatic radar only
    python scripts/train.py --channels multi   # monostatic + bistatic receiver

Writes the ensemble weights to results/models/ and the predictions for every
split (in physical units, with total / aleatoric / epistemic variance) to
results/predictions_<channels>.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar.model import GaitNet, beta_nll, ensemble_predict  # noqa: E402

TARGETS = ["speed", "cadence"]
SPLITS = ["train", "val", "test", "oblique", "low_snr"]


def load(path: str, channels: str):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    x = d["mono"][:, None] if channels == "mono" else np.stack([d["mono"], d["bi"]], axis=1)
    y = np.array([[m[t] for t in TARGETS] for m in meta], dtype=np.float32)
    split = np.array([m["split"] for m in meta])
    return x, y, split, meta


def augment(xb: torch.Tensor, crop: int, gen: torch.Generator) -> torch.Tensor:
    """Random time crop + exact Doppler mirror (labels are invariant to both)."""
    n_t = xb.shape[-1]
    s = int(torch.randint(0, n_t - crop + 1, (1,), generator=gen))
    xb = xb[..., s:s + crop]
    if torch.rand(1, generator=gen) < 0.5:
        xb = doppler_flip(xb)
    return xb


def doppler_flip(xb: torch.Tensor) -> torch.Tensor:
    """Mirror the Doppler axis exactly around 0 Hz.

    The Doppler axis is f_i = (i - N/2) * df for i = 0..N-1 (N even), so -f_i
    sits at index N - i. A plain flip would map i -> N-1-i, i.e. f -> -f - df
    (a one-bin bias); flipping and rolling by one bin maps i -> N - i exactly.
    Bin 0 (-N/2 df) has no positive counterpart and is kept in place.
    """
    return torch.roll(torch.flip(xb, dims=[2]), shifts=1, dims=2)


def train_member(seed, x_tr, y_tr, x_va, y_va, in_ch, epochs, lr, batch, crop):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = GaitNet(in_ch=in_ch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(x_tr) / batch)))
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = augment(x_tr[idx], crop, gen)
            mu, lv = model(xb)
            loss = beta_nll(mu, lv, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            mu, lv = model(x_va)
            val = float((0.5 * (lv + (y_va - mu) ** 2 / lv.exp())).mean())
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  seed {seed} epoch {ep + 1:2d}/{epochs}  val NLL {val:+.3f}", flush=True)
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gait_dataset.npz")
    ap.add_argument("--channels", choices=["mono", "multi"], default="mono")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=112, help="training time-crop (frames of 122)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    x, y, split, _ = load(a.data, a.channels)
    tr, va = split == "train", split == "val"
    y_mean, y_std = y[tr].mean(0), y[tr].std(0)
    yz = (y - y_mean) / y_std
    X = torch.from_numpy(x.astype(np.float32))
    Y = torch.from_numpy(yz)

    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    models = []
    t0 = time.time()
    for k in range(a.members):
        m, best = train_member(1000 + k, X[tr], Y[tr], X[va], Y[va], x.shape[1], a.epochs, a.lr, a.batch, a.crop)
        torch.save(m.state_dict(), out / "models" / f"{a.channels}_member{k}.pt")
        models.append(m)
        print(f"member {k}: best val NLL {best:+.3f}  ({time.time() - t0:.0f} s)", flush=True)

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
    np.savez(out / f"predictions_{a.channels}.npz", **res)
    print(f"saved predictions to {out / f'predictions_{a.channels}.npz'}")


if __name__ == "__main__":
    main()
