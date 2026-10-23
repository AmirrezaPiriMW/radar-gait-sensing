"""Generate the simulated radar gait dataset.

Splits are *subject-disjoint*: train, validation and test share no person (the
oblique and low-SNR sets reuse the six test people):

    train   : 28 subjects x 36 walks, nominal conditions
    val     :  6 subjects x 24 walks, nominal conditions
    test    :  6 subjects x 30 walks, nominal conditions
    oblique :  test subjects x 20 walks, heading 45-70 deg (unseen range)
    low_snr :  test subjects x 20 walks, SNR -22 to -14 dB (unseen range)

Usage:  python scripts/generate_dataset.py --out data/gait_dataset.npz
        python scripts/generate_dataset.py --decoupled --out data/gait_dataset_decoupled.npz

With ``--decoupled`` the same people and splits are used, but speed and cadence
are drawn independently (within plausible step lengths) instead of following
the speed-coupled Boulic cadence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar.dataset import make_recording, make_subjects  # noqa: E402
from gaitradar.radar import RadarConfig  # noqa: E402

N_SUBJECTS = 40
SPLITS = {
    # name: (subject ids, walks per subject, condition)
    "train": (range(0, 28), 36, "nominal"),
    "val": (range(28, 34), 24, "nominal"),
    "test": (range(34, 40), 30, "nominal"),
    "oblique": (range(34, 40), 20, "oblique"),
    "low_snr": (range(34, 40), 20, "low_snr"),
}


def _job(args):
    split, sid, k, cond, seed, decoupled = args
    subjects = make_subjects(N_SUBJECTS, seed)
    split_id = list(SPLITS).index(split)
    key = [seed, split_id, sid, k] + ([1] if decoupled else [])
    rng = np.random.default_rng(key)
    rec = make_recording(subjects[sid], cond, rng, RadarConfig(), decoupled=decoupled)
    rec.meta["split"] = split
    return rec.spec_mono.astype(np.float16), rec.spec_bi.astype(np.float16), rec.meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gait_dataset.npz")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of walks per subject (quick runs)")
    ap.add_argument("--decoupled", action="store_true",
                    help="draw speed and cadence independently (within plausible step lengths)")
    a = ap.parse_args()

    jobs = []
    for split, (sids, n, cond) in SPLITS.items():
        n = max(2, int(round(n * a.scale)))
        jobs += [(split, sid, k, cond, a.seed, a.decoupled) for sid in sids for k in range(n)]

    t0 = time.time()
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(_job, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} walks  ({time.time() - t0:.0f} s)", flush=True)

    mono = np.stack([r[0] for r in res])
    bi = np.stack([r[1] for r in res])
    meta = [r[2] for r in res]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mono=mono, bi=bi, meta=json.dumps(meta),
                        subjects=json.dumps([s.to_dict() for s in make_subjects(N_SUBJECTS, a.seed)]))
    print(f"saved {len(meta)} walks to {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
