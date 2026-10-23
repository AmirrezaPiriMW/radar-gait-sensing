"""Evaluate the physics baseline and the CNN ensembles; write metrics and figures.

    python scripts/evaluate.py [--uq-model multi]

Reads data/gait_dataset.npz and results/predictions_{mono,multi}.npz and
writes results/metrics.json, results/metrics.md and figures/fig2-fig5.

Protocol
--------
* All numbers are computed on subject-disjoint splits (6 unseen people).
* Uncertainty is recalibrated post hoc with one variance scale per target,
  fitted on the validation split only (never on test data). Every reported
  sigma, interval and coverage uses the recalibrated variance unless a field
  name ends in ``_raw``.
* 95 % confidence intervals use a *subject-level cluster bootstrap*: people
  are resampled with replacement and all of their walks are kept together,
  because walks of the same person are not independent. Differences between
  methods are computed on the same resampled walks (paired).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar import uq  # noqa: E402
from gaitradar.baseline import estimate_cadence, estimate_speed  # noqa: E402
from gaitradar.radar import RadarConfig  # noqa: E402

TARGETS = ["speed", "cadence"]
UNITS = {"speed": "m/s", "cadence": "steps/min"}
EVAL_SPLITS = ["test", "oblique", "low_snr"]
SPLIT_LABEL = {"test": "Nominal", "oblique": "Oblique heading (45–70°)", "low_snr": "Low SNR (−22 to −14 dB)"}
SPLIT_COL = {"test": "#1f4e79", "oblique": "#e07b39", "low_snr": "#9b2d30"}
METHODS = ["baseline", "mono", "multi"]
COL = {"baseline": "#8c8c8c", "mono": "#e07b39", "multi": "#1f4e79"}
NAME = {"baseline": "Physics baseline (monostatic)", "mono": "CNN ensemble, monostatic",
        "multi": "CNN ensemble, multistatic"}
N_BOOT = 2000

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


# --------------------------------------------------------------------------- statistics
def cluster_bootstrap(subjects: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = 0):
    """95 % CI of ``stat(idx)`` under resampling of subjects (clusters)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    members = {s: np.flatnonzero(subjects == s) for s in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[s] for s in pick])
        vals[b] = stat(idx)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "data/gait_dataset.npz"))
    ap.add_argument("--results", default=str(root / "results"))
    ap.add_argument("--figures", default=str(root / "figures"))
    ap.add_argument("--uq-model", default="multi", choices=["mono", "multi"],
                    help="ensemble shown in the uncertainty figures")
    a = ap.parse_args()
    d = np.load(a.data)
    meta = json.loads(str(d["meta"]))
    cfg = RadarConfig()
    figs, res_dir = Path(a.figures), Path(a.results)
    figs.mkdir(parents=True, exist_ok=True)

    # ---------------- predictions of all three methods, aligned per split ----------------
    # All statistics in float64 (network outputs are stored as float32).
    preds = {c: {k: (v.astype(np.float64) if v.dtype.kind == "f" else v) for k, v in np.load(res_dir / f"predictions_{c}.npz").items()}
             for c in ("mono", "multi")}
    scale = {c: [uq.fit_sigma_scale(p["val_y"][:, j], p["val_mu"][:, j], np.sqrt(p["val_var"][:, j]))
                 for j in range(2)] for c, p in preds.items()}
    R = {}
    for s in EVAL_SPLITS:
        idx = preds["mono"][f"{s}_index"]
        assert np.array_equal(idx, preds["multi"][f"{s}_index"])
        y = np.array([[meta[i]["speed"], meta[i]["cadence"]] for i in idx])
        for c in ("mono", "multi"):
            assert np.allclose(preds[c][f"{s}_y"], y, atol=1e-4)
        spec = d["mono"][idx].astype(np.float32)
        base_mu = np.array([[estimate_speed(x, cfg), estimate_cadence(x, cfg)] for x in spec])
        R[s] = dict(idx=idx, y=y, subj=np.array([meta[i]["subject_id"] for i in idx]),
                    los=np.array([meta[i]["los_aspect_deg"] for i in idx]),
                    mu={"baseline": base_mu, "mono": preds["mono"][f"{s}_mu"], "multi": preds["multi"][f"{s}_mu"]},
                    sig={c: np.sqrt(preds[c][f"{s}_var"]) * np.array(scale[c]) for c in ("mono", "multi")},
                    sig_raw={c: np.sqrt(preds[c][f"{s}_var"]) for c in ("mono", "multi")})

    # ---------------- metrics ----------------
    metrics = {"recalibration_scale": scale, "n_walks": {s: int(len(R[s]["idx"])) for s in EVAL_SPLITS},
               "n_subjects": {s: int(len(np.unique(R[s]["subj"]))) for s in EVAL_SPLITS}}
    for s in EVAL_SPLITS:
        r = R[s]
        metrics[s] = {m: {} for m in METHODS}
        metrics[s]["paired_differences"] = {}
        for j, t in enumerate(TARGETS):
            err = {m: np.abs(r["mu"][m][:, j] - r["y"][:, j]) for m in METHODS}
            for m in METHODS:
                e = err[m]
                entry = dict(mae=float(e.mean()), mae_ci95=cluster_bootstrap(r["subj"], lambda i, e=e: e[i].mean()),
                             rmse=uq.rmse(r["y"][:, j], r["mu"][m][:, j]))
                if m != "baseline":
                    sig, sraw = r["sig"][m][:, j], r["sig_raw"][m][:, j]
                    cal = uq.summary(r["y"][:, j], r["mu"][m][:, j], sig)
                    raw = uq.summary(r["y"][:, j], r["mu"][m][:, j], sraw)
                    k, sel = uq.selective_mae(r["y"][:, j], r["mu"][m][:, j], sig, keep=np.array([0.8]))
                    p = preds[m]
                    entry.update({k_: v for k_, v in cal.items() if k_ not in ("mae", "rmse")})
                    entry.update(coverage90_raw=raw["coverage90"], cal_error_raw=raw["cal_error"],
                                 mean_sigma_raw=raw["mean_sigma"], mae_keep80=float(sel[0]),
                                 aleatoric_rms=float(np.sqrt(np.mean(p[f"{s}_aleatoric"][:, j]))) * scale[m][j],
                                 epistemic_rms=float(np.sqrt(np.mean(p[f"{s}_epistemic"][:, j]))) * scale[m][j])
                metrics[s][m][t] = entry
            diffs = {}
            for a_, b_ in (("baseline", "mono"), ("mono", "multi"), ("baseline", "multi")):
                dlt = err[a_] - err[b_]
                diffs[f"{a_}_minus_{b_}"] = dict(mean=float(dlt.mean()),
                                                 ci95=cluster_bootstrap(r["subj"], lambda i, dlt=dlt: dlt[i].mean()))
            metrics[s]["paired_differences"][t] = diffs

    # Speed MAE by line-of-sight angle (nominal + oblique walks), as plotted in Fig. 2.
    los_all = np.concatenate([R[s]["los"] for s in ("test", "oblique")])
    metrics["speed_mae_by_los_bin"] = {}
    for m_ in METHODS:
        e_all = np.concatenate([np.abs(R[s]["mu"][m_][:, 0] - R[s]["y"][:, 0]) for s in ("test", "oblique")])
        rows = []
        for lo_ in range(0, 90, 10):
            sel_ = (los_all >= lo_) & (los_all < lo_ + 10)
            if sel_.sum() >= 8:
                rows.append(dict(lo=lo_, hi=lo_ + 10, n=int(sel_.sum()), mae=float(e_all[sel_].mean())))
        metrics["speed_mae_by_los_bin"][m_] = rows

    # Pooled selective prediction (all three conditions mixed, as in deployment).
    metrics["selective_pooled"] = {}
    for m in ("mono", "multi"):
        y = np.concatenate([R[s]["y"][:, 0] for s in EVAL_SPLITS])
        mu = np.concatenate([R[s]["mu"][m][:, 0] for s in EVAL_SPLITS])
        sig = np.concatenate([R[s]["sig"][m][:, 0] for s in EVAL_SPLITS])
        keep, sel = uq.selective_mae(y, mu, sig, keep=np.array([1.0, 0.8, 0.6]))
        metrics["selective_pooled"][m] = dict(mae_all=float(sel[0]), mae_keep80=float(sel[1]), mae_keep60=float(sel[2]),
                                              spearman=uq.error_uncertainty_corr(y, mu, sig))

    # Gait-prior check: speed predicted from the TRUE cadence and height alone (no radar).
    sp = np.array([m["speed"] for m in meta]); cad = np.array([m["cadence"] for m in meta])
    hgt = np.array([m["height"] for m in meta]); spl = np.array([m["split"] for m in meta])
    A = np.column_stack([cad, cad * hgt, hgt, np.ones_like(hgt)])
    coef = np.linalg.lstsq(A[spl == "train"], sp[spl == "train"], rcond=None)[0]
    metrics["gait_prior_speed_mae"] = {s: float(np.abs(A[spl == s] @ coef - sp[spl == s]).mean()) for s in EVAL_SPLITS}
    metrics["speed_cadence_corr"] = float(np.corrcoef(sp, cad)[0, 1])
    # If a network leans on the speed-cadence coupling, its speed errors should
    # correlate with the errors of the cadence+height-only predictor.
    prior_pred = A @ coef
    metrics["prior_error_correlation"] = {}
    for m_ in ("mono", "multi"):
        metrics["prior_error_correlation"][m_] = {}
        for s in EVAL_SPLITS:
            idx = R[s]["idx"]
            rc = R[s]["mu"][m_][:, 0] - R[s]["y"][:, 0]
            rp = prior_pred[idx] - sp[idx]
            rr = float(np.corrcoef(rc, rp)[0, 1])
            ci = cluster_bootstrap(R[s]["subj"], lambda i, rc=rc, rp=rp: np.corrcoef(rc[i], rp[i])[0, 1])
            metrics["prior_error_correlation"][m_][s] = dict(r=rr, ci95=ci)
    los_tr = np.array([m["los_aspect_deg"] for m in meta])[spl == "train"]
    metrics["train_los_aspect"] = dict(frac_above_40=float(np.mean(los_tr > 40)), p95=float(np.percentile(los_tr, 95)))

    # Uncertainty-gated fallback (monostatic): switch to the physics baseline when the
    # CNN's speed sigma exceeds the 95th percentile of its validation sigmas.
    sig_va = np.sqrt(preds["mono"]["val_var"][:, 0]) * scale["mono"][0]
    thr = float(np.quantile(sig_va, 0.95))
    metrics["gated_fallback_mono"] = {"threshold_sigma": thr}
    for s in EVAL_SPLITS:
        r = R[s]
        sig = r["sig"]["mono"][:, 0]
        g = np.where(sig > thr, r["mu"]["baseline"][:, 0], r["mu"]["mono"][:, 0])
        metrics["gated_fallback_mono"][s] = dict(cnn=uq.mae(r["y"][:, 0], r["mu"]["mono"][:, 0]),
                                                 baseline=uq.mae(r["y"][:, 0], r["mu"]["baseline"][:, 0]),
                                                 gated=uq.mae(r["y"][:, 0], g),
                                                 fallback_fraction=float(np.mean(sig > thr)))

    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_markdown(metrics, res_dir / "metrics.md")
    print((res_dir / "metrics.md").read_text())

    prior = float(np.mean([metrics["gait_prior_speed_mae"][k] for k in ("test", "oblique")]))
    fig_speed_vs_aspect(R, los_tr, prior, figs / "fig2_speed_vs_aspect.png")
    fig_pred_vs_true(R, a.uq_model, figs / "fig3_pred_vs_true.png")
    fig_calibration(R, a.uq_model, figs / "fig4_calibration.png")
    fig_trust(R, a.uq_model, figs / "fig5_trust.png")


# --------------------------------------------------------------------------- report
def _ci(e, nd):
    return f"{e['mae']:.{nd}f} [{e['mae_ci95'][0]:.{nd}f}, {e['mae_ci95'][1]:.{nd}f}]"


def write_markdown(m, path):
    nd = {"speed": 3, "cadence": 2}
    L = ["# Results", "",
         "MAE with 95 % subject-level cluster-bootstrap CI in brackets. Uncertainty columns use the "
         "variance recalibrated on validation data.", ""]
    for t in TARGETS:
        L += [f"## {t.capitalize()} ({UNITS[t]})", "",
              "| Condition | Physics baseline | CNN mono | CNN multi |", "|---|---|---|---|"]
        for s in EVAL_SPLITS:
            L.append(f"| {SPLIT_LABEL[s]} | " + " | ".join(_ci(m[s][k][t], nd[t]) for k in METHODS) + " |")
        L += ["", "| Condition | Model | 90 % interval coverage | calib. error | Spearman(abs err, σ) | mean σ | MAE, 80 % most confident |",
              "|---|---|---|---|---|---|---|"]
        for s in EVAL_SPLITS:
            for k in ("mono", "multi"):
                e = m[s][k][t]
                L.append(f"| {SPLIT_LABEL[s]} | {k} | {100 * e['coverage90']:.1f} % | {e['cal_error']:.3f} | "
                         f"{e['spearman']:.2f} | {e['mean_sigma']:.{nd[t]}f} | {e['mae_keep80']:.{nd[t]}f} |")
        L += ["", "Paired differences in MAE (positive = second method better):", "",
              "| Condition | baseline − mono | mono − multi | baseline − multi |", "|---|---|---|---|"]
        for s in EVAL_SPLITS:
            pdiff = m[s]["paired_differences"][t]
            cells = [f"{pdiff[k]['mean']:+.{nd[t]}f} [{pdiff[k]['ci95'][0]:+.{nd[t]}f}, {pdiff[k]['ci95'][1]:+.{nd[t]}f}]"
                     for k in ("baseline_minus_mono", "mono_minus_multi", "baseline_minus_multi")]
            L.append(f"| {SPLIT_LABEL[s]} | " + " | ".join(cells) + " |")
        L.append("")
    sp = m["selective_pooled"]
    L += ["## Pooled selective prediction (speed, all conditions)", "",
          "| Model | MAE all | MAE keep 80 % | MAE keep 60 % | Spearman |", "|---|---|---|---|---|"]
    for k in ("mono", "multi"):
        L.append(f"| {k} | {sp[k]['mae_all']:.3f} | {sp[k]['mae_keep80']:.3f} | {sp[k]['mae_keep60']:.3f} | {sp[k]['spearman']:.2f} |")
    g = m["gated_fallback_mono"]
    L += ["", f"## Uncertainty-gated fallback (mono CNN → baseline when σ > {g['threshold_sigma']:.3f} m/s)", "",
          "| Condition | CNN | baseline | gated | fallback share |", "|---|---|---|---|---|"]
    for s in EVAL_SPLITS:
        e = g[s]
        L.append(f"| {SPLIT_LABEL[s]} | {e['cnn']:.3f} | {e['baseline']:.3f} | {e['gated']:.3f} | {e['fallback_fraction']:.2f} |")
    L += ["", "Speed from true cadence + height only (no radar), MAE: "
          + ", ".join(f"{SPLIT_LABEL[s]} {m['gait_prior_speed_mae'][s]:.3f}" for s in EVAL_SPLITS)
          + f"; speed–cadence correlation r = {m['speed_cadence_corr']:.2f}.",
          f"Training walks with line-of-sight angle > 40°: {100 * m['train_los_aspect']['frac_above_40']:.0f} %.", "",
          "## Correlation of CNN speed errors with cadence+height-prior errors", "",
          "| Condition | mono r [95 % CI] | multi r [95 % CI] |", "|---|---|---|"]
    for s in EVAL_SPLITS:
        cells = [f"{m['prior_error_correlation'][k][s]['r']:.2f} [{m['prior_error_correlation'][k][s]['ci95'][0]:.2f}, "
                 f"{m['prior_error_correlation'][k][s]['ci95'][1]:.2f}]" for k in ("mono", "multi")]
        L.append(f"| {SPLIT_LABEL[s]} | " + " | ".join(cells) + " |")
    L += [""]
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------- figures
def fig_speed_vs_aspect(R, los_train, prior_mae, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax2 = ax.twinx()
    hb = np.arange(0, 91, 5)
    w = np.ones_like(los_train) / len(los_train)
    ax2.hist(los_train, bins=hb, weights=w, color="#dbe3ec", alpha=0.7, zorder=0)
    ax2.set_ylabel("share of training walks", color="#7f8c9a")
    ax2.tick_params(axis="y", colors="#7f8c9a"); ax2.grid(False)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color("#c9d3de")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    bins = np.arange(0, 91, 10)
    for key in METHODS:
        asp = np.concatenate([R[s]["los"] for s in ("test", "oblique")])
        err = np.concatenate([np.abs(R[s]["mu"][key][:, 0] - R[s]["y"][:, 0]) for s in ("test", "oblique")])
        b = np.digitize(asp, bins) - 1
        centers, means, lo, hi = [], [], [], []
        rng = np.random.default_rng(1)
        for k in range(len(bins) - 1):
            e = err[b == k]
            if len(e) >= 8:
                centers.append(bins[k] + 5); means.append(e.mean())
                boot = [rng.choice(e, len(e)).mean() for _ in range(500)]
                lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
        ax.plot(centers, means, "o-", color=COL[key], label=NAME[key], lw=1.8, ms=4)
        ax.fill_between(centers, lo, hi, color=COL[key], alpha=0.15, lw=0)
    ax.axhline(prior_mae, color="k", ls=":", lw=1, label="speed from true cadence + height (no radar)")
    ax.set_xlabel("Angle between walking direction and radar line of sight at mid-walk (deg)")
    ax.set_ylabel("Speed MAE (m/s)")
    ax.set_title("Speed error vs. viewing angle (nominal + oblique walks, unseen people; bands: 95 % CI)", fontsize=8.5)
    ax.set_xlim(0, 90)
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="none", fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_pred_vs_true(R, model, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    z = norm.ppf(0.95)
    marker = {"test": "o", "oblique": "s", "low_snr": "^"}
    for j, (t, ax) in enumerate(zip(TARGETS, axes)):
        for s in EVAL_SPLITS:
            r = R[s]
            ax.errorbar(r["y"][:, j], r["mu"][model][:, j], yerr=z * r["sig"][model][:, j], fmt=marker[s], ms=2.8,
                        color=SPLIT_COL[s], ecolor=SPLIT_COL[s], elinewidth=0.5, alpha=0.55, label=SPLIT_LABEL[s])
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", lw=0.8); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"True {t} ({UNITS[t]})"); ax.set_ylabel(f"Predicted {t} ({UNITS[t]})")
        ax.set_title(f"{t.capitalize()}: prediction ± 90 % interval", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"{NAME[model]}, unseen people", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_calibration(R, model, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for j, (t, ax) in enumerate(zip(TARGETS, axes)):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        for s in EVAL_SPLITS:
            r = R[s]
            lv, cov = uq.calibration_curve(r["y"][:, j], r["mu"][model][:, j], r["sig"][model][:, j])
            ax.plot(lv, cov, "-o", ms=3, color=SPLIT_COL[s], label=SPLIT_LABEL[s])
            if s == "test":
                lv, cov = uq.calibration_curve(r["y"][:, j], r["mu"][model][:, j], r["sig_raw"][model][:, j])
                ax.plot(lv, cov, ":", color=SPLIT_COL[s], lw=1.2, label="Nominal, before recalibration")
        ax.set_xlabel("Nominal central-interval coverage"); ax.set_ylabel("Observed coverage")
        ax.set_title(t.capitalize(), fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Interval calibration, {NAME[model]} (variance scale fitted on validation data)", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_trust(R, model, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for j, t in enumerate(TARGETS):
        ax = axes[j]
        bp = ax.boxplot([R[s]["sig"][model][:, j] for s in EVAL_SPLITS], patch_artist=True, widths=0.55,
                        showfliers=False, medianprops=dict(color="k"))
        for patch, s in zip(bp["boxes"], EVAL_SPLITS):
            patch.set_facecolor(SPLIT_COL[s]); patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], ["Nominal", "Oblique", "Low SNR"])
        ax.set_ylabel(f"Predicted σ ({UNITS[t]})")
        ax.set_title(f"{t.capitalize()}: predicted σ by condition", fontsize=9)
    ax = axes[2]
    y = np.concatenate([R[s]["y"][:, 0] for s in EVAL_SPLITS])
    mu = np.concatenate([R[s]["mu"][model][:, 0] for s in EVAL_SPLITS])
    sig = np.concatenate([R[s]["sig"][model][:, 0] for s in EVAL_SPLITS])
    keep, sel = uq.selective_mae(y, mu, sig)
    ax.plot(keep * 100, sel, "-", color="#1f4e79", lw=1.8, label="reject highest predicted σ first")
    rng = np.random.default_rng(0)
    rand = np.mean([uq.selective_mae(y, mu, rng.random(len(y)))[1] for _ in range(200)], axis=0)
    ax.plot(keep * 100, rand, "--", color="#8c8c8c", lw=1.2, label="random rejection")
    ax.invert_xaxis()
    ax.set_xlabel("Walks kept (%)"); ax.set_ylabel("Speed MAE (m/s)")
    ax.set_title("Selective prediction, all conditions pooled", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(NAME[model], fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
