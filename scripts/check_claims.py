"""Check that every number quoted in the documents matches the computed metrics.

Expected strings are generated from results/metrics.json and
results/decoupled/metrics_decoupled.json (never typed by hand) and must
appear verbatim in the given documents. Run with the text of each document:

    python scripts/check_claims.py --readme README.md [--deck deck.txt] [--cv cv.txt] [--letter letter.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results/metrics.json").read_text())
MD = json.loads((ROOT / "results/decoupled/metrics_decoupled.json").read_text())
S = ["test", "oblique", "low_snr"]


def n(x, nd):
    """Number formatted like the documents: typographic minus for negatives."""
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")


def sn(x, nd):
    """Signed number (+/−)."""
    return ("+" if x >= 0 else "−") + f"{abs(x):.{nd}f}"


def ci(v, lo, hi, nd):
    return f"{n(v, nd)} [{n(lo, nd)}, {n(hi, nd)}]"


def readme_claims():
    c = []
    nd = {"speed": 3, "cadence": 2}
    for s in S:
        for t in ("speed", "cadence"):
            for m in ("baseline", "mono", "multi"):
                e = M[s][m][t]
                c.append((f"main table {s}/{t}/{m}", ci(e["mae"], *e["mae_ci95"], nd[t])))
    for data in ("C", "D"):
        for s in S:
            for m in ("baseline", "mono_C", "multi_C", "mono_D", "multi_D"):
                e = MD["mae"][data][s]["speed"][m]
                c.append((f"decoupling table {data}/{s}/{m}", ci(e["mae"], *e["ci95"], 3)))
    sp = lambda s, m: M[s][m]["speed"]["mae"]
    c += [("ratio nominal", f"{sp('test', 'baseline') / sp('test', 'mono'):.1f}×"),
          ("ratio oblique", f"{sp('oblique', 'baseline') / sp('oblique', 'mono'):.1f}×")]
    b = M["speed_mae_by_los_bin"]
    c += [("baseline < 20 deg", f"{max(r['mae'] for r in b['baseline'] if r['hi'] <= 20):.2f} m/s".replace(" ", " ")),
          ("baseline > 70 deg", f"{max(r['mae'] for r in b['baseline'] if r['lo'] >= 70):.2f} m/s")]
    lo20 = [r["mae"] for k in ("mono", "multi") for r in b[k] if r["hi"] <= 20]
    lo60 = [r["mae"] for k in ("mono", "multi") for r in b[k] if r["hi"] <= 60]
    c += [("CNN < 20 deg range", f"{min(lo20):.2f}–{max(lo20):.2f}"), ("CNN < 60 deg range", f"{min(lo60):.2f}–{max(lo60):.2f}")]
    pd = M["low_snr"]["paired_differences"]["speed"]["baseline_minus_mono"]
    c.append(("low-SNR paired", f"{n(pd['mean'], 2)} m/s, CI [{n(pd['ci95'][0], 2)}, {n(pd['ci95'][1], 2)}]"))
    pd = M["test"]["paired_differences"]["cadence"]["baseline_minus_mono"]
    c.append(("cadence nominal paired CI", f"[{sn(pd['ci95'][0], 2)}, {sn(pd['ci95'][1], 2)}]"))
    pd = M["oblique"]["paired_differences"]["cadence"]["mono_minus_multi"]
    c.append(("cadence oblique mono-multi", f"{sn(pd['mean'], 2)}, CI [{sn(pd['ci95'][0], 2)}, {sn(pd['ci95'][1], 2)}]"))
    c.append(("cadence oblique values", f"{M['oblique']['mono']['cadence']['mae']:.2f} to {M['oblique']['multi']['cadence']['mae']:.2f}"))
    c.append(("speed-cadence r", f"r = {M['speed_cadence_corr']:.2f}"))
    g = M["gait_prior_speed_mae"]
    c.append(("prior MAE range", f"{min(g.values()):.2f}–{max(g.values()):.2f} m/s"))
    pe = M["prior_error_correlation"]
    for s, lab in (("test", "nominal"), ("oblique", "oblique"), ("low_snr", "low SNR")):
        v = [pe["mono"][s]["r"], pe["multi"][s]["r"]]
        c.append((f"prior corr {lab}", f"{min(v):.2f}–{max(v):.2f}"))
    mm = M
    c += [("sigma nominal", f"{mm['test']['multi']['speed']['mean_sigma']:.3f} (nominal)"),
          ("sigma oblique", f"{mm['oblique']['multi']['speed']['mean_sigma']:.3f} (oblique)"),
          ("sigma low SNR", f"{mm['low_snr']['multi']['speed']['mean_sigma']:.3f} m/s (low SNR)"),
          ("pooled Spearman", f"Spearman {M['selective_pooled']['multi']['spearman']:.2f}"),
          ("selective 80", f"from {M['selective_pooled']['multi']['mae_all']:.3f} to {M['selective_pooled']['multi']['mae_keep80']:.3f} m/s"),
          ("selective 60", f"{M['selective_pooled']['multi']['mae_keep60']:.3f} m/s (Fig. 5)")]
    cov = lambda s, t: f"{100 * M[s]['multi'][t]['coverage90']:.1f} %"
    c += [("coverage speed nominal", f"cover {cov('test', 'speed')} of nominal"), ("coverage speed oblique", f"{cov('oblique', 'speed')} of oblique"),
          ("coverage speed low", f"{cov('low_snr', 'speed')} of low-SNR"),
          ("coverage cadence", f"(cadence: {cov('test', 'cadence')}, {cov('oblique', 'cadence')}, {cov('low_snr', 'cadence')}")]
    gf = M["gated_fallback_mono"]
    c += [("fallback low SNR", f"{gf['low_snr']['cnn']:.3f} → {gf['low_snr']['gated']:.3f} m/s"),
          ("fallback oblique", f"{gf['oblique']['cnn']:.3f} → {gf['oblique']['gated']:.3f} m/s"),
          ("train LOS > 40", f"{100 * M['train_los_aspect']['frac_above_40']:.0f} % of training walks")]
    # decoupling findings
    d = lambda data, s, m: MD["mae"][data][s]["speed"][m]["mae"]
    P = MD["paired"]
    c += [("decoupled r", f"from {MD['speed_cadence_corr']['C']:.2f} to {MD['speed_cadence_corr']['D']:.2f}")]
    gD = MD["gait_prior_speed_mae"]["D"]
    c.append(("decoupled prior MAE", f"to {min(gD.values()):.2f}–{max(gD.values()):.2f} m/s"))
    pdn = P["D"]["test"]["speed"]["mono_C_minus_mono_D"]
    c.append(("finding 7 mono", f"{d('D', 'test', 'mono_C'):.3f} vs. {d('D', 'test', 'mono_D'):.3f} m/s for the decoupled-trained model (paired difference {sn(pdn['mean'], 3)}, CI [{sn(pdn['ci95'][0], 3)}, {sn(pdn['ci95'][1], 3)}])"))
    pdm = P["D"]["test"]["speed"]["multi_C_minus_multi_D"]
    c.append(("finding 7 multi", f"{d('D', 'test', 'multi_C'):.3f} vs. {d('D', 'test', 'multi_D'):.3f} m/s ({sn(pdm['mean'], 3)}, CI [{sn(pdm['ci95'][0], 3)}, {sn(pdm['ci95'][1], 3)}])"))
    c.append(("finding 7 baseline", f"physics baseline ({d('D', 'test', 'baseline'):.3f} m/s)"))
    rr = [MD["prior_error_corr"]["D"][s][m]["r"] for s in S for m in ("mono_C", "multi_C")]
    c.append(("finding 7 corr", f"r = {min(rr):.2f}–{max(rr):.2f}"))
    cv = MD["coverage90"]["D"]["test"]["speed"]
    c.append(("finding 7 coverage", f"only {100 * cv['mono_C']:.1f} % (mono) and {100 * cv['multi_C']:.1f} % (multi)"))
    c.append(("finding 7 coverage D", f"against {100 * cv['mono_D']:.1f} % for"))
    ms = MD["mean_sigma_raw_speed"]
    up = lambda new, old: round(100 * (new / old - 1))
    c.append(("finding 7 error rise", f"rises by {up(d('D', 'test', 'mono_C'), d('C', 'test', 'mono_C'))} % (mono) and {up(d('D', 'test', 'multi_C'), d('C', 'test', 'multi_C'))} % (multi)"))
    c.append(("finding 7 sigma rise", f"rises by only {up(ms['D']['test']['mono_C'], ms['C']['test']['mono_C'])} % and {up(ms['D']['test']['multi_C'], ms['C']['test']['multi_C'])} %"))
    cc = MD["mae"]["D"]["test"]["cadence"]
    c.append(("finding 7 cadence", f"{cc['mono_C']['mae']:.2f} and {cc['multi_C']['mae']:.2f} steps/min, against {cc['mono_D']['mae']:.2f} and {cc['multi_D']['mae']:.2f}"))
    p1 = P["C"]["test"]["speed"]["mono_C_minus_mono_D"]; p2 = P["C"]["test"]["speed"]["multi_C_minus_multi_D"]
    c.append(("finding 8 typical mono", f"mono {d('C', 'test', 'mono_D'):.3f} vs. {d('C', 'test', 'mono_C'):.3f} m/s, paired CI [{sn(p1['ci95'][0], 3)}, {sn(p1['ci95'][1], 3)}]"))
    c.append(("finding 8 typical multi", f"multi {d('C', 'test', 'multi_D'):.3f} vs. {d('C', 'test', 'multi_C'):.3f} m/s, paired CI [{sn(p2['ci95'][0], 3)}, {sn(p2['ci95'][1], 3)}]"))
    po = P["D"]["oblique"]["speed"]["mono_D_minus_multi_D"]; pc = P["C"]["oblique"]["speed"]["mono_D_minus_multi_D"]
    c.append(("finding 8 multistatic D", f"from {d('D', 'oblique', 'mono_D'):.3f} to {d('D', 'oblique', 'multi_D'):.3f} m/s on decoupled data (paired {sn(po['mean'], 3)}, CI [{sn(po['ci95'][0], 3)}, {sn(po['ci95'][1], 3)}])"))
    c.append(("finding 8 multistatic C", f"from {d('C', 'oblique', 'mono_D'):.3f} to {d('C', 'oblique', 'multi_D'):.3f} m/s on typical gait ({sn(pc['mean'], 3)}, CI [{sn(pc['ci95'][0], 3)}, {sn(pc['ci95'][1], 3)}])"))
    gains = [P[x]["test"]["speed"]["mono_D_minus_multi_D"]["mean"] for x in ("C", "D")]
    c.append(("finding 8 nominal gain", f"(≤ {max(gains):.3f} m/s)"))
    pe8 = P["C"]["oblique"]["speed"]["mono_C_minus_mono_D"]
    c.append(("finding 8 exception", f"({d('C', 'oblique', 'mono_D'):.3f} vs. {d('C', 'oblique', 'mono_C'):.3f} m/s; paired {sn(pe8['mean'], 3)}, CI [{sn(pe8['ci95'][0], 3)}, {sn(pe8['ci95'][1], 3)}])"))
    pl = P["C"]["low_snr"]["speed"]["multi_C_minus_multi_D"]
    c.append(("finding 8 low SNR", f"{d('C', 'low_snr', 'multi_C'):.3f} → {d('C', 'low_snr', 'multi_D'):.3f} m/s, paired {sn(pl['mean'], 3)}, CI [{sn(pl['ci95'][0], 3)}, {sn(pl['ci95'][1], 3)}]"))
    c.append(("finding 8 baseline low SNR", f"far better there ({d('C', 'low_snr', 'baseline'):.3f} m/s)"))
    # five of six claim
    learned = ("mono_C", "multi_C", "mono_D", "multi_D")
    best = sum(min(learned, key=lambda m: d(x, s, m)) == "multi_D" for x in ("C", "D") for s in S)
    gap = max(d(x, s, "multi_D") - min(d(x, s, m) for m in learned) for x in ("C", "D") for s in S)
    c.append(("finding 8 five of six", f"in {['zero', 'one', 'two', 'three', 'four', 'five', 'six'][best]} of the six test conditions and is within {gap:.3f} m/s"))
    cvD = MD["coverage90"]["D"]
    ob = sorted([100 * cvD["oblique"]["speed"][m] for m in ("mono_D", "multi_D")]); lw = sorted([100 * cvD["low_snr"]["speed"][m] for m in ("mono_D", "multi_D")])
    c.append(("finding 9 coverage", f"only {ob[0]:.1f}–{ob[1]:.1f} % of oblique and {lw[0]:.1f}–{lw[1]:.1f} % of low-SNR"))
    rl = sorted([MD["prior_error_corr"]["D"]["low_snr"][m]["r"] for m in ("mono_D", "multi_D")])
    c.append(("finding 9 residual corr", f"r = {rl[0]:.2f}–{rl[1]:.2f}"))
    pb = P["D"]["test"]["cadence"]["baseline_minus_multi_D"]
    c.append(("finding 9 cadence", f"{MD['mae']['D']['test']['cadence']['baseline']['mae']:.2f} vs. {MD['mae']['D']['test']['cadence']['multi_D']['mae']:.2f} steps/min for the decoupled multistatic model; paired CI [{n(pb['ci95'][0], 2)}, {n(pb['ci95'][1], 2)}]"))
    return c


def cv_letter_claims():
    sp = lambda s, m: M[s][m]["speed"]["mae"]
    gain = 1 - MD["mae"]["D"]["oblique"]["speed"]["multi_D"]["mae"] / MD["mae"]["D"]["oblique"]["speed"]["mono_D"]["mae"]
    cv = [("CV speed", f"{sp('test', 'mono'):.3f} m/s"), ("CV baseline", f"{sp('test', 'baseline'):.3f} m/s"),
          ("CV ratio nominal", f"({sp('test', 'baseline') / sp('test', 'mono'):.1f}×)"),
          ("CV ratio oblique", f"{sp('oblique', 'baseline') / sp('oblique', 'mono'):.1f}× lower"),
          ("CV multistatic gain", f"{round(100 * gain)} %")]
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--deck"); ap.add_argument("--cv"); ap.add_argument("--letter")
    a = ap.parse_args()
    norm = lambda t: t.replace("\u00a0", " ").replace("\u2060", "").replace("**", "")
    fails, total = 0, 0
    jobs = [("README", a.readme, readme_claims())]
    if a.cv:
        jobs.append(("CV", a.cv, cv_letter_claims()))
    for doc, path, claims in jobs:
        text = norm(Path(path).read_text())
        for name, exp in claims:
            total += 1
            if norm(exp) not in text:
                fails += 1
                print(f"MISSING in {doc}: {name}: '{exp}'")
    print(f"{total - fails}/{total} document claims verified")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
