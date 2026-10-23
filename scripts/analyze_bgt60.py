"""Process real Infineon BGT60 recordings with the physics baseline.

    python scripts/analyze_bgt60.py --raw rec1/radar.npy rec2/radar.npy --config rec1/config.json \
        [--reference data/bgt60_reference.csv] [--t-chirp 200e-6 ...] --out results/bgt60

Outputs, per recording: a micro-Doppler figure, the estimated radial speed
and cadence, and (if a reference table is given) the errors against the
reference. Reference CSV columns (one row per recording):

    file, distance_m, duration_s, steps, heading_deg

speed = distance_m / duration_s, cadence = 60 * steps / duration_s. heading_deg
(optional, angle between walking direction and radar boresight) is used to
report the radial component a single monostatic radar can see; this correction
is only approximate, because the line-of-sight angle changes along a walk that
is not aimed at the radar. Walks straight towards or away from the radar give
the cleanest speed reference. Rows without a reference are processed but not
scored. Files are matched to reference rows by file name.

Note: the CNN ensembles were trained on simulated 5.8 GHz spectrograms, so they
are not applied here. Using them on 60 GHz data needs the simulator re-run
with the BGT60 configuration (see gaitradar.bgt60.simulate_bgt60_recording).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar.baseline import estimate_cadence, estimate_speed, torso_doppler  # noqa: E402
from gaitradar.bgt60 import BGT60Config, config_from_json, load_raw, micro_doppler_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="radar .npy files (frames, rx, chirps, samples)")
    ap.add_argument("--config", help="recording config JSON (Radar Fusion GUI / Radar SDK)")
    ap.add_argument("--reference", help="CSV with file, distance_m, duration_s, steps, heading_deg")
    for f in ("f_start", "f_end", "sample_rate", "t_chirp", "t_frame"):
        ap.add_argument("--" + f.replace("_", "-"), type=float, help=f"override {f}")
    ap.add_argument("--out", default="results/bgt60")
    a = ap.parse_args()

    over = {f: getattr(a, f) for f in ("f_start", "f_end", "sample_rate", "t_chirp", "t_frame") if getattr(a, f) is not None}
    cfg = config_from_json(a.config, **over) if a.config else BGT60Config(**over)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"fc {cfg.fc / 1e9:.2f} GHz, B {cfg.bandwidth / 1e9:.2f} GHz, range res {100 * cfg.range_resolution:.1f} cm, "
          f"max unambiguous speed {cfg.max_unambiguous_speed:.2f} m/s, frame rate {cfg.frame_rate:.1f} Hz")
    for w in cfg.check():
        print("WARNING:", w)
    if getattr(cfg, "missing_keys", None):
        print("WARNING: not found in config, defaults used for:", ", ".join(cfg.missing_keys))

    ref = {}
    if a.reference:
        with open(a.reference, newline="") as fh:
            for row in csv.DictReader(fh):
                ref[Path(row["file"]).name] = row          # matched by file name

    rows = []
    for p in a.raw:
        raw = load_raw(p)
        cfg.n_chirps, cfg.n_samples = raw.shape[2], raw.shape[3]
        spec = micro_doppler_frames(cfg, raw)
        v_rad, cad = estimate_speed(spec, cfg), estimate_cadence(spec, cfg)
        row = dict(file=str(p), frames=raw.shape[0], duration_s=raw.shape[0] * cfg.t_frame,
                   radial_speed_mps=v_rad, cadence_spm=cad)
        r = ref.get(Path(p).name)
        if r:
            v_true = float(r["distance_m"]) / float(r["duration_s"])
            hd = float(r["heading_deg"]) if r.get("heading_deg") not in (None, "") else 0.0
            row.update(true_speed_mps=v_true, true_radial_speed_mps=v_true * np.cos(np.deg2rad(hd)),
                       speed_error_mps=v_rad - v_true * np.cos(np.deg2rad(hd)))
            if r.get("steps") not in (None, ""):
                c_true = 60 * float(r["steps"]) / float(r["duration_s"])
                row.update(true_cadence_spm=c_true, cadence_error_spm=cad - c_true)
        rows.append(row)

        fd = cfg.doppler_axis
        t = np.arange(spec.shape[1]) * cfg.t_frame
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.imshow(spec, aspect="auto", origin="lower", cmap="jet",
                  extent=[t[0], t[-1], fd[0] * cfg.wavelength / 2, fd[-1] * cfg.wavelength / 2])
        ax.plot(t, torso_doppler(spec, cfg) * cfg.wavelength / 2, "w.", ms=2)
        ax.set_xlabel("time (s)"); ax.set_ylabel("radial velocity (m/s)")
        ax.set_title(f"{Path(p).parent.name}/{Path(p).name}: radial speed {v_rad:.2f} m/s, cadence {cad:.0f} steps/min", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"{Path(p).parent.name}_{Path(p).stem}.png", dpi=150); plt.close(fig)

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "estimates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    scored = [r for r in rows if "speed_error_mps" in r]
    summary = dict(n_recordings=len(rows), n_scored=len(scored))
    if scored:
        summary["radial_speed_mae_mps"] = float(np.mean([abs(r["speed_error_mps"]) for r in scored]))
        cs = [r for r in scored if "cadence_error_spm" in r]
        if cs:
            summary["cadence_mae_spm"] = float(np.mean([abs(r["cadence_error_spm"]) for r in cs]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
