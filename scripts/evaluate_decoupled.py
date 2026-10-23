"""Decoupling experiment: do the networks rely on the speed-cadence coupling?

Two model families are trained on two datasets that differ only in how speed
and cadence are sampled:

* **C** (coupled): cadence follows the Boulic relation (speed-cadence r ~ 0.9),
* **D** (decoupled): speed and cadence drawn independently within plausible
  step lengths (r ~ 0.6, the minimum the physics of walking allows here).

Every family is evaluated on both kinds of test data (same six unseen people).
If a family leans on the coupling, its speed error should rise when the
coupling is removed at test time. Comparing mono and multi within family D
tests whether the second receiver helps once the shortcut is gone.

    python scripts/evaluate_decoupled.py

Requires results/predictions_*.npz, results/decoupled/predictions_*.npz and
results/cross/*.npz (see run_all.sh). Writes results/decoupled/metrics_decoupled.{json,md}
and figures/fig6_decoupled.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import cluster_bootstrap  # noqa: E402
from gaitradar import uq  # noqa: E402
from gaitradar.baseline import estimate_cadence, estimate_speed  # noqa: E402
from gaitradar.radar import RadarConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["test", "oblique", "low_snr"]
SPLIT_LABEL = {"test": "Nominal", "oblique": "Oblique heading", "low_snr": "Low SNR"}
DATA = {"C": ROOT / "data/gait_dataset.npz", "D": ROOT / "data/gait_dataset_decoupled.npz"}
METHODS = ["baseline", "mono_C", "multi_C", "mono_D", "multi_D"]
NAME = {"baseline": "Physics baseline", "mono_C": "Mono, trained coupled", "multi_C": "Multi, trained coupled",
        "mono_D": "Mono, trained decoupled", "multi_D": "Multi, trained decoupled"}
COL = {"baseline": "#8c8c8c", "mono_C": "#f2b48a", "multi_C": "#8fb0cf", "mono_D": "#e07b39", "multi_D": "#1f4e79"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def pred_file(family: str, ch: str, data: str) -> Path:
    if family == data:
        return (ROOT / "results" / ("decoupled" if family == "D" else "") / f"predictions_{ch}.npz")
    src = "coupled" if family == "C" else "decoupled"
    dst = "decoupled" if data == "D" else "coupled"
    return ROOT / "results/cross" / f"{src}_{ch}_on_{dst}.npz"


def load_pred(path: Path) -> dict:
    """Load predictions with float arrays promoted to float64."""
    raw = np.load(path)
    return {k: (raw[k].astype(np.float64) if raw[k].dtype.kind == "f" else raw[k]) for k in raw.files}


def own_scale(family: str, ch: str):
    p = load_pred(pred_file(family, ch, family))
    return [uq.fit_sigma_scale(p["val_y"][:, j], p["val_mu"][:, j], np.sqrt(p["val_var"][:, j])) for j in range(2)]


def main():
    cfg = RadarConfig()
    out = {"speed_cadence_corr": {}, "gait_prior_speed_mae": {}}
    R = {}
    for data in ("C", "D"):
        d = np.load(DATA[data])
        meta = json.loads(str(d["meta"]))
        sp = np.array([m["speed"] for m in meta]); cad = np.array([m["cadence"] for m in meta])
        hgt = np.array([m["height"] for m in meta]); spl = np.array([m["split"] for m in meta])
        out["speed_cadence_corr"][data] = float(np.corrcoef(sp[spl == "train"], cad[spl == "train"])[0, 1])
        # Prior check: speed from TRUE cadence + height, fitted on this dataset's training split.
        A = np.column_stack([cad, cad * hgt, hgt, np.ones_like(hgt)])
        coef = np.linalg.lstsq(A[spl == "train"], sp[spl == "train"], rcond=None)[0]
        prior = A @ coef
        out["gait_prior_speed_mae"][data] = {s: float(np.abs(prior[spl == s] - sp[spl == s]).mean()) for s in SPLITS}
        R[data] = {}
        for s in SPLITS:
            idx = np.flatnonzero(spl == s)
            spec = d["mono"][idx].astype(np.float32)
            mu = {"baseline": np.array([[estimate_speed(x, cfg), estimate_cadence(x, cfg)] for x in spec])}
            sig = {}
            for fam in ("C", "D"):
                for ch in ("mono", "multi"):
                    p = load_pred(pred_file(fam, ch, data))
                    assert np.array_equal(p[f"{s}_index"], idx)
                    assert np.allclose(p[f"{s}_y"], np.column_stack([sp[idx], cad[idx]]), atol=1e-4)
                    mu[f"{ch}_{fam}"] = p[f"{s}_mu"]
                    sig[f"{ch}_{fam}"] = np.sqrt(p[f"{s}_var"]) * np.array(own_scale(fam, ch))
            R[data][s] = dict(idx=idx, y=np.column_stack([sp[idx], cad[idx]]), mu=mu, sig=sig, prior=prior[idx],
                              subj=np.array([meta[i]["subject_id"] for i in idx]),
                              los=np.array([meta[i]["los_aspect_deg"] for i in idx]))

    # MAE with subject-level cluster-bootstrap CIs; paired differences of interest.
    PAIRS = [("mono_C", "mono_D"), ("multi_C", "multi_D"), ("mono_D", "multi_D"), ("baseline", "multi_D")]
    out["mae"], out["paired"], out["coverage90"], out["prior_error_corr"] = {}, {}, {}, {}
    # Mean raw (un-recalibrated) predicted speed sigma: does the uncertainty react to the shift?
    out["mean_sigma_raw_speed"] = {data: {s: {m: float(np.mean(R[data][s]["sig"][m][:, 0] / own_scale(m.split("_")[1], m.split("_")[0])[0]))
                                              for m in ("mono_C", "multi_C", "mono_D", "multi_D")} for s in SPLITS} for data in ("C", "D")}
    for data in ("C", "D"):
        out["mae"][data], out["paired"][data], out["coverage90"][data], out["prior_error_corr"][data] = {}, {}, {}, {}
        for s in SPLITS:
            r = R[data][s]
            out["mae"][data][s], out["paired"][data][s], out["coverage90"][data][s] = {}, {}, {}
            out["prior_error_corr"][data][s] = {}
            for j, t in enumerate(("speed", "cadence")):
                err = {m: np.abs(r["mu"][m][:, j] - r["y"][:, j]) for m in METHODS}
                out["mae"][data][s][t] = {m: dict(mae=float(err[m].mean()),
                                                  ci95=cluster_bootstrap(r["subj"], lambda i, e=err[m]: e[i].mean()))
                                          for m in METHODS}
                out["paired"][data][s][t] = {}
                for a_, b_ in PAIRS:
                    dl = err[a_] - err[b_]
                    out["paired"][data][s][t][f"{a_}_minus_{b_}"] = dict(
                        mean=float(dl.mean()), ci95=cluster_bootstrap(r["subj"], lambda i, dl=dl: dl[i].mean()))
                out["coverage90"][data][s][t] = {m: uq.coverage(r["y"][:, j], r["mu"][m][:, j], r["sig"][m][:, j], 0.9)
                                                 for m in METHODS if m != "baseline"}
            rp = r["prior"] - r["y"][:, 0]
            for m in ("mono_C", "multi_C", "mono_D", "multi_D"):
                rc = r["mu"][m][:, 0] - r["y"][:, 0]
                out["prior_error_corr"][data][s][m] = dict(
                    r=float(np.corrcoef(rc, rp)[0, 1]),
                    ci95=cluster_bootstrap(r["subj"], lambda i, rc=rc, rp=rp: np.corrcoef(rc[i], rp[i])[0, 1]))

    res_dir = ROOT / "results/decoupled"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "metrics_decoupled.json").write_text(json.dumps(out, indent=2))
    write_markdown(out, res_dir / "metrics_decoupled.md")
    print((res_dir / "metrics_decoupled.md").read_text())
    figure(R, out, ROOT / "figures/fig6_decoupled.png")


def write_markdown(o, path):
    nd = {"speed": 3, "cadence": 2}
    L = ["# Decoupling experiment", "",
         f"Speed–cadence correlation in the training data: coupled r = {o['speed_cadence_corr']['C']:.2f}, "
         f"decoupled r = {o['speed_cadence_corr']['D']:.2f}.", "",
         "Speed from true cadence + height only (no radar), MAE (m/s): "
         + "; ".join(f"{'coupled' if k == 'C' else 'decoupled'} " + ", ".join(f"{SPLIT_LABEL[s]} {v[s]:.3f}" for s in SPLITS)
                     for k, v in o["gait_prior_speed_mae"].items()), "",
         "MAE with 95 % subject-level cluster-bootstrap CI. C = trained on coupled data, D = trained on decoupled data.", ""]
    for t in ("speed", "cadence"):
        for data in ("C", "D"):
            L += [f"## {t.capitalize()} on {'coupled' if data == 'C' else 'decoupled'} test data", "",
                  "| Condition | " + " | ".join(NAME[m] for m in METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
            for s in SPLITS:
                cells = [f"{o['mae'][data][s][t][m]['mae']:.{nd[t]}f} [{o['mae'][data][s][t][m]['ci95'][0]:.{nd[t]}f}, "
                         f"{o['mae'][data][s][t][m]['ci95'][1]:.{nd[t]}f}]" for m in METHODS]
                L.append(f"| {SPLIT_LABEL[s]} | " + " | ".join(cells) + " |")
            L += ["", "Paired differences (positive = second method better):", "",
                  "| Condition | mono C − mono D | multi C − multi D | mono D − multi D | baseline − multi D |", "|---|---|---|---|---|"]
            for s in SPLITS:
                pd = o["paired"][data][s][t]
                cells = [f"{pd[k]['mean']:+.{nd[t]}f} [{pd[k]['ci95'][0]:+.{nd[t]}f}, {pd[k]['ci95'][1]:+.{nd[t]}f}]"
                         for k in ("mono_C_minus_mono_D", "multi_C_minus_multi_D", "mono_D_minus_multi_D", "baseline_minus_multi_D")]
                L.append(f"| {SPLIT_LABEL[s]} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## Correlation of speed errors with cadence+height-prior errors", "",
          "| Test data | Condition | mono C | multi C | mono D | multi D |", "|---|---|---|---|---|---|"]
    for data in ("C", "D"):
        for s in SPLITS:
            pe = o["prior_error_corr"][data][s]
            cells = [f"{pe[m]['r']:.2f} [{pe[m]['ci95'][0]:.2f}, {pe[m]['ci95'][1]:.2f}]" for m in ("mono_C", "multi_C", "mono_D", "multi_D")]
            L.append(f"| {'coupled' if data == 'C' else 'decoupled'} | {SPLIT_LABEL[s]} | " + " | ".join(cells) + " |")
    L += ["", "## Mean raw predicted speed sigma (m/s)", "", "| Test data | Condition | mono C | multi C | mono D | multi D |", "|---|---|---|---|---|---|"]
    for data in ("C", "D"):
        for s in SPLITS:
            ms = o["mean_sigma_raw_speed"][data][s]
            L.append(f"| {'coupled' if data == 'C' else 'decoupled'} | {SPLIT_LABEL[s]} | " + " | ".join(f"{ms[m]:.3f}" for m in ("mono_C", "multi_C", "mono_D", "multi_D")) + " |")
    L += ["", "## 90 % interval coverage (each family recalibrated on its own validation data)", "",
          "| Test data | Condition | target | mono C | multi C | mono D | multi D |", "|---|---|---|---|---|---|---|"]
    for data in ("C", "D"):
        for s in SPLITS:
            for t in ("speed", "cadence"):
                cv = o["coverage90"][data][s][t]
                L.append(f"| {'coupled' if data == 'C' else 'decoupled'} | {SPLIT_LABEL[s]} | {t} | "
                         + " | ".join(f"{100 * cv[m]:.1f} %" for m in ("mono_C", "multi_C", "mono_D", "multi_D")) + " |")
    path.write_text("\n".join(L) + "\n")


def figure(R, o, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    w = 0.16
    xs = np.arange(len(SPLITS))
    for k, m in enumerate(METHODS):
        vals = [o["mae"]["D"][s]["speed"][m]["mae"] for s in SPLITS]
        lo = [v - o["mae"]["D"][s]["speed"][m]["ci95"][0] for v, s in zip(vals, SPLITS)]
        hi = [o["mae"]["D"][s]["speed"][m]["ci95"][1] - v for v, s in zip(vals, SPLITS)]
        ax.bar(xs + (k - 2) * w, vals, w, color=COL[m], label=NAME[m], yerr=[lo, hi], capsize=2,
               error_kw=dict(lw=0.8, ecolor="#333333"))
    prior = [o["gait_prior_speed_mae"]["D"][s] for s in SPLITS]
    for x, p in zip(xs, prior):
        ax.hlines(p, x - 2.6 * w, x + 2.6 * w, colors="k", linestyles=":", lw=1)
    ax.plot([], [], "k:", lw=1, label="speed from true cadence + height")
    ax.set_xticks(xs, [SPLIT_LABEL[s] for s in SPLITS])
    ax.set_ylabel("Speed MAE (m/s)")
    ax.set_title("Speed on decoupled test data (unseen people; 95 % CI)", fontsize=9)
    ax.set_ylim(0, 0.56)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    bins = np.arange(0, 91, 15)
    for m in ("baseline", "mono_D", "multi_D"):
        los = np.concatenate([R["D"][s]["los"] for s in ("test", "oblique")])
        err = np.concatenate([np.abs(R["D"][s]["mu"][m][:, 0] - R["D"][s]["y"][:, 0]) for s in ("test", "oblique")])
        b = np.digitize(los, bins) - 1
        c, mm = [], []
        for k in range(len(bins) - 1):
            if (b == k).sum() >= 8:
                c.append(bins[k] + 7.5); mm.append(err[b == k].mean())
        ax.plot(c, mm, "o-", color=COL[m], label=NAME[m], lw=1.8, ms=4)
    ax.set_xlabel("Angle to radar line of sight at mid-walk (deg)")
    ax.set_ylabel("Speed MAE (m/s)")
    ax.set_title("Decoupled test data: error vs. viewing angle", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
