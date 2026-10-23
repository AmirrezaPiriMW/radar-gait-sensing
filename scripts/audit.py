"""Independent audit of the datasets, predictions and reported metrics.

Recomputes the reported numbers with separate, deliberately simple code (no
imports from scripts/evaluate*.py or gaitradar.uq) and compares them with
results/metrics.json and results/decoupled/metrics_decoupled.json.

    python scripts/audit.py            # exits with status 1 if any check fails
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaitradar.baseline import estimate_cadence, estimate_speed  # noqa: E402
from gaitradar.radar import RadarConfig  # noqa: E402

SPLIT_N = {"train": 1008, "val": 144, "test": 180, "oblique": 120, "low_snr": 120}
SPLIT_SUBJ = {"train": set(range(0, 28)), "val": set(range(28, 34)), "test": set(range(34, 40)),
              "oblique": set(range(34, 40)), "low_snr": set(range(34, 40))}
EVAL = ["test", "oblique", "low_snr"]
results = []
_BASE = {}


def baseline_estimates(d, path_tag: str, idx: np.ndarray, cfg) -> np.ndarray:
    """Physics-baseline (speed, cadence) per walk, cached per dataset and split."""
    key = (path_tag, tuple(idx[:3]), len(idx))
    if key not in _BASE:
        _BASE[key] = np.array([[estimate_speed(d["mono"][i].astype(np.float32), cfg),
                                estimate_cadence(d["mono"][i].astype(np.float32), cfg)] for i in idx])
    return _BASE[key]


def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---------------------------------------------------------------- independent metric code
def mae(y, m):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(m))))


def coverage90(y, m, s):
    return float(np.mean(np.abs(y - m) <= norm.ppf(0.95) * s))


def var_scale(y, m, s):
    return float(np.sqrt(np.mean(((y - m) / s) ** 2)))


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(subj, stat, n=2000, seed=12345):
    """Cluster bootstrap over subjects, independent RNG stream from evaluate.py."""
    rng = np.random.default_rng(seed)
    u = np.unique(subj)
    groups = [np.flatnonzero(subj == s) for s in u]
    v = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def ci_close(ci_rep, ci_new, width_tol=0.25):
    """CIs from different bootstrap streams must agree to within 25 % of the CI width."""
    w = max(ci_rep[1] - ci_rep[0], 1e-12)
    return abs(ci_rep[0] - ci_new[0]) <= width_tol * w and abs(ci_rep[1] - ci_new[1]) <= width_tol * w


# ---------------------------------------------------------------- data integrity
def audit_dataset(path: Path, tag: str, decoupled: bool):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    mono, bi = d["mono"], d["bi"]
    check(f"[{tag}] spectrogram shape (N, 192, 122) for both channels",
          mono.shape == bi.shape == (len(meta), 192, 122), str(mono.shape))
    check(f"[{tag}] spectrograms finite and within [0, 1]",
          np.isfinite(mono).all() and np.isfinite(bi).all() and mono.min() >= 0 and mono.max() <= 1 and bi.min() >= 0 and bi.max() <= 1)
    spl = np.array([m["split"] for m in meta])
    for s, n in SPLIT_N.items():
        check(f"[{tag}] split '{s}' has {n} walks", (spl == s).sum() == n, str((spl == s).sum()))
        subj = {m["subject_id"] for m in meta if m["split"] == s}
        check(f"[{tag}] split '{s}' uses exactly the intended subjects", subj == SPLIT_SUBJ[s])
    for a in ("train", "val"):
        for b in ("val", "test"):
            if a != b:
                ov = SPLIT_SUBJ[a] & {m["subject_id"] for m in meta if m["split"] == b}
                check(f"[{tag}] no subject shared between '{a}' and '{b}'", not ov)
    sp = np.array([m["speed"] for m in meta]); cad = np.array([m["cadence"] for m in meta])
    sl = np.array([m["step_length"] for m in meta]); h = np.array([m["height"] for m in meta])
    check(f"[{tag}] step_length = 60 * speed / cadence for every walk", np.allclose(sl, 60 * sp / cad, rtol=1e-9))
    check(f"[{tag}] speeds within 0.5-1.8 m/s (+-3 % variability)", sp.min() >= 0.5 * 0.95 and sp.max() <= 1.8 * 1.05,
          f"{sp.min():.3f}-{sp.max():.3f}")
    alpha = np.abs([m["alpha_deg"] for m in meta]); snr = np.array([m["snr_db"] for m in meta])
    for s in SPLIT_N:
        k = spl == s
        lo, hi = (45, 70) if s == "oblique" else (0, 40)
        check(f"[{tag}] '{s}' headings within {lo}-{hi} deg", alpha[k].min() >= lo - 1e-9 and alpha[k].max() <= hi + 1e-9)
        slo, shi = (-22, -14) if s == "low_snr" else (-6, 8)
        check(f"[{tag}] '{s}' SNR within {slo} to {shi} dB", snr[k].min() >= slo and snr[k].max() <= shi)
    if decoupled:
        rel = sl / h
        check(f"[{tag}] every walk marked as decoupled", all(m.get("sampling") == "decoupled" for m in meta))
        check(f"[{tag}] step length 0.22-0.50 x height (+-3 % variability)", rel.min() >= 0.22 * 0.94 and rel.max() <= 0.50 * 1.06,
              f"{rel.min():.3f}-{rel.max():.3f}")
    else:
        check(f"[{tag}] no walk marked as decoupled", all("sampling" not in m for m in meta))
    r = np.corrcoef(sp[spl == "train"], cad[spl == "train"])[0, 1]
    check(f"[{tag}] training speed-cadence correlation {'< 0.70' if decoupled else '> 0.85'}",
          (r < 0.70) if decoupled else (r > 0.85), f"r = {r:.3f}")
    return d, meta


def load_pred(path: Path) -> dict:
    """Load a predictions file with every array promoted to float64."""
    raw = np.load(path)
    return {k: (raw[k].astype(np.float64) if raw[k].dtype.kind == "f" else raw[k]) for k in raw.files}


def audit_predictions(path: Path, meta, tag: str):
    p = load_pred(path)
    y_true = np.array([[m["speed"], m["cadence"]] for m in meta])
    spl = np.array([m["split"] for m in meta])
    for s in SPLIT_N:
        idx = p[f"{s}_index"]
        check(f"[{tag}] '{s}' indices match the dataset split", np.array_equal(idx, np.flatnonzero(spl == s)))
        check(f"[{tag}] '{s}' labels match metadata", np.allclose(p[f"{s}_y"], y_true[idx], atol=1e-4))
        check(f"[{tag}] '{s}' variances finite and positive",
              np.isfinite(p[f"{s}_var"]).all() and (p[f"{s}_var"] > 0).all() and np.isfinite(p[f"{s}_mu"]).all())
        check(f"[{tag}] '{s}' total variance = aleatoric + epistemic",
              np.allclose(p[f"{s}_var"], p[f"{s}_aleatoric"] + p[f"{s}_epistemic"], rtol=1e-5))
    return p


# ---------------------------------------------------------------- main experiment
def audit_main(d, meta):
    m = json.loads((ROOT / "results/metrics.json").read_text())
    cfg = RadarConfig()
    spl = np.array([x["split"] for x in meta])
    P = {c: load_pred(ROOT / f"results/predictions_{c}.npz") for c in ("mono", "multi")}
    scale = {c: [var_scale(P[c]["val_y"][:, j], P[c]["val_mu"][:, j], np.sqrt(P[c]["val_var"][:, j])) for j in range(2)]
             for c in P}
    for c in P:
        for j in range(2):
            check(f"[main] recalibration scale {c}/{['speed', 'cadence'][j]}", close(scale[c][j], m["recalibration_scale"][c][j]))
    pooled = {c: ([], [], []) for c in P}
    for s in EVAL:
        idx = np.flatnonzero(spl == s)
        subj = np.array([meta[i]["subject_id"] for i in idx])
        y = np.array([[meta[i]["speed"], meta[i]["cadence"]] for i in idx])
        base = baseline_estimates(d, "C", idx, cfg)
        for j, t in enumerate(("speed", "cadence")):
            e_b = np.abs(base[:, j] - y[:, j])
            check(f"[main] {s} {t} baseline MAE", close(e_b.mean(), m[s]["baseline"][t]["mae"]),
                  f"{e_b.mean():.4f} vs {m[s]['baseline'][t]['mae']:.4f}")
            check(f"[main] {s} {t} baseline CI", ci_close(m[s]["baseline"][t]["mae_ci95"], boot_ci(subj, lambda i: e_b[i].mean())))
            errs = {"baseline": e_b}
            for c in P:
                mu, sg = P[c][f"{s}_mu"][:, j], np.sqrt(P[c][f"{s}_var"][:, j]) * scale[c][j]
                e = np.abs(mu - y[:, j]); errs[c] = e
                rep = m[s][c][t]
                check(f"[main] {s} {t} {c} MAE", close(e.mean(), rep["mae"]), f"{e.mean():.4f}")
                check(f"[main] {s} {t} {c} MAE CI", ci_close(rep["mae_ci95"], boot_ci(subj, lambda i, e=e: e[i].mean())))
                check(f"[main] {s} {t} {c} 90 % coverage", close(coverage90(y[:, j], mu, sg), rep["coverage90"]))
                check(f"[main] {s} {t} {c} mean sigma", close(sg.mean(), rep["mean_sigma"]))
                check(f"[main] {s} {t} {c} Spearman", close(spearman(e, sg), rep["spearman"], 1e-6))
                if j == 0:
                    pooled[c][0].append(y[:, 0]); pooled[c][1].append(mu); pooled[c][2].append(sg)
            for a_, b_ in (("baseline", "mono"), ("mono", "multi"), ("baseline", "multi")):
                dl = errs[a_] - errs[b_]
                rep = m[s]["paired_differences"][t][f"{a_}_minus_{b_}"]
                check(f"[main] {s} {t} paired {a_}-{b_} mean", close(dl.mean(), rep["mean"]))
                check(f"[main] {s} {t} paired {a_}-{b_} CI", ci_close(rep["ci95"], boot_ci(subj, lambda i, dl=dl: dl[i].mean())))
    for c in P:
        y, mu, sg = (np.concatenate(v) for v in pooled[c])
        o = np.argsort(sg); e = np.abs(y - mu)[o]
        rep = m["selective_pooled"][c]
        check(f"[main] pooled selective {c}: all", close(e.mean(), rep["mae_all"]))
        check(f"[main] pooled selective {c}: keep 80 %", close(e[: int(round(0.8 * len(e)))].mean(), rep["mae_keep80"]))
        check(f"[main] pooled selective {c}: keep 60 %", close(e[: int(round(0.6 * len(e)))].mean(), rep["mae_keep60"]))
        check(f"[main] pooled Spearman {c}", close(spearman(np.abs(y - mu), sg), rep["spearman"], 1e-6))
    # Gated fallback (mono)
    thr = np.quantile(np.sqrt(P["mono"]["val_var"][:, 0]) * scale["mono"][0], 0.95)
    check("[main] fallback threshold", close(thr, m["gated_fallback_mono"]["threshold_sigma"]))
    for s in EVAL:
        idx = np.flatnonzero(spl == s)
        y = np.array([meta[i]["speed"] for i in idx])
        b = baseline_estimates(d, "C", idx, cfg)[:, 0]
        mu, sg = P["mono"][f"{s}_mu"][:, 0], np.sqrt(P["mono"][f"{s}_var"][:, 0]) * scale["mono"][0]
        g = np.where(sg > thr, b, mu)
        check(f"[main] fallback {s} gated MAE", close(mae(y, g), m["gated_fallback_mono"][s]["gated"]))
        check(f"[main] fallback {s} share", close(np.mean(sg > thr), m["gated_fallback_mono"][s]["fallback_fraction"]))
    # Gait prior
    sp = np.array([x["speed"] for x in meta]); cad = np.array([x["cadence"] for x in meta]); h = np.array([x["height"] for x in meta])
    X = np.column_stack([cad, cad * h, h, np.ones_like(h)])
    w = np.linalg.solve(X[spl == "train"].T @ X[spl == "train"], X[spl == "train"].T @ sp[spl == "train"])
    for s in EVAL:
        check(f"[main] gait-prior MAE {s}", close(mae(sp[spl == s], X[spl == s] @ w), m["gait_prior_speed_mae"][s], 1e-6))
        for c in P:
            idx = np.flatnonzero(spl == s)
            r = np.corrcoef(P[c][f"{s}_mu"][:, 0] - sp[idx], X[idx] @ w - sp[idx])[0, 1]
            check(f"[main] prior-error correlation {s} {c}", close(r, m["prior_error_correlation"][c][s]["r"], 1e-6))
    los_tr = np.array([x["los_aspect_deg"] for x in meta])[spl == "train"]
    check("[main] share of training walks with LOS > 40 deg", close(np.mean(los_tr > 40), m["train_los_aspect"]["frac_above_40"]))


# ---------------------------------------------------------------- decoupling experiment
def audit_decoupled(dC, metaC, dD, metaD):
    path = ROOT / "results/decoupled/metrics_decoupled.json"
    if not path.exists():
        check("[decoupled] metrics file exists", False, str(path)); return
    o = json.loads(path.read_text())
    cfg = RadarConfig()
    files = {("C", "C"): "results/predictions_{ch}.npz", ("D", "D"): "results/decoupled/predictions_{ch}.npz",
             ("C", "D"): "results/cross/coupled_{ch}_on_decoupled.npz", ("D", "C"): "results/cross/decoupled_{ch}_on_coupled.npz"}
    own = {}
    for fam in ("C", "D"):
        for ch in ("mono", "multi"):
            p = load_pred(ROOT / files[(fam, fam)].format(ch=ch))
            own[(fam, ch)] = [var_scale(p["val_y"][:, j], p["val_mu"][:, j], np.sqrt(p["val_var"][:, j])) for j in range(2)]
    for data, (d, meta) in (("C", (dC, metaC)), ("D", (dD, metaD))):
        spl = np.array([x["split"] for x in meta])
        for fam in ("C", "D"):
            for ch in ("mono", "multi"):
                audit_predictions(ROOT / files[(fam, data)].format(ch=ch), meta, f"pred {ch}_{fam} on {data}")
        for s in EVAL:
            idx = np.flatnonzero(spl == s)
            subj = np.array([meta[i]["subject_id"] for i in idx])
            y = np.array([[meta[i]["speed"], meta[i]["cadence"]] for i in idx])
            base = baseline_estimates(d, data, idx, cfg)
            for j, t in enumerate(("speed", "cadence")):
                errs = {"baseline": np.abs(base[:, j] - y[:, j])}
                for fam in ("C", "D"):
                    for ch in ("mono", "multi"):
                        p = load_pred(ROOT / files[(fam, data)].format(ch=ch))
                        mu = p[f"{s}_mu"][:, j]; sg = np.sqrt(p[f"{s}_var"][:, j]) * own[(fam, ch)][j]
                        errs[f"{ch}_{fam}"] = np.abs(mu - y[:, j])
                        if j == 0:
                            check(f"[decoupled] {data}/{s} {ch}_{fam} mean raw sigma",
                                  close(np.sqrt(p[f"{s}_var"][:, 0]).mean(), o["mean_sigma_raw_speed"][data][s][f"{ch}_{fam}"], 1e-6))
                        check(f"[decoupled] {data}/{s}/{t} {ch}_{fam} coverage",
                              close(coverage90(y[:, j], mu, sg), o["coverage90"][data][s][t][f"{ch}_{fam}"]))
                for k, e in errs.items():
                    rep = o["mae"][data][s][t][k]
                    check(f"[decoupled] {data}/{s}/{t} {k} MAE", close(e.mean(), rep["mae"]), f"{e.mean():.4f}")
                    check(f"[decoupled] {data}/{s}/{t} {k} CI", ci_close(rep["ci95"], boot_ci(subj, lambda i, e=e: e[i].mean())))
                for a_, b_ in (("mono_C", "mono_D"), ("multi_C", "multi_D"), ("mono_D", "multi_D"), ("baseline", "multi_D")):
                    dl = errs[a_] - errs[b_]
                    rep = o["paired"][data][s][t][f"{a_}_minus_{b_}"]
                    check(f"[decoupled] {data}/{s}/{t} paired {a_}-{b_}", close(dl.mean(), rep["mean"]))
                    check(f"[decoupled] {data}/{s}/{t} paired {a_}-{b_} CI", ci_close(rep["ci95"], boot_ci(subj, lambda i, dl=dl: dl[i].mean())))
        sp = np.array([x["speed"] for x in meta]); cad = np.array([x["cadence"] for x in meta]); h = np.array([x["height"] for x in meta])
        X = np.column_stack([cad, cad * h, h, np.ones_like(h)])
        w = np.linalg.solve(X[spl == "train"].T @ X[spl == "train"], X[spl == "train"].T @ sp[spl == "train"])
        check(f"[decoupled] training correlation {data}", close(np.corrcoef(sp[spl == "train"], cad[spl == "train"])[0, 1],
                                                               o["speed_cadence_corr"][data], 1e-6))
        for s in EVAL:
            check(f"[decoupled] gait-prior MAE {data}/{s}", close(mae(sp[spl == s], X[spl == s] @ w), o["gait_prior_speed_mae"][data][s], 1e-6))


def main():
    dC, metaC = audit_dataset(ROOT / "data/gait_dataset.npz", "coupled data", decoupled=False)
    for c in ("mono", "multi"):
        audit_predictions(ROOT / f"results/predictions_{c}.npz", metaC, f"pred {c}_C on C")
    audit_main(dC, metaC)
    pD = ROOT / "data/gait_dataset_decoupled.npz"
    if pD.exists():
        dD, metaD = audit_dataset(pD, "decoupled data", decoupled=True)
        if (ROOT / "results/decoupled/predictions_multi.npz").exists():
            audit_decoupled(dC, metaC, dD, metaD)
        else:
            check("[decoupled] multistatic predictions available", False, "training not finished")
    n_ok = sum(r[0] for r in results)
    for ok, name, det in results:
        if not ok:
            print(f"FAIL  {name}  {det}")
    print(f"\n{n_ok}/{len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
