"""Figure 1: the signal chain, from body model to micro-Doppler and cadence.

    python scripts/plot_signatures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar.baseline import (CLUTTER_NOTCH_HZ, cadence_velocity_diagram,  # noqa: E402
                                estimate_cadence, estimate_speed, torso_doppler)
from gaitradar.dataset import make_recording  # noqa: E402
from gaitradar.kinematics import (SCATTERER_NAMES, SCATTERER_WEIGHTS, Subject,  # noqa: E402
                                  body_frame_positions, gait_params)
from gaitradar.radar import RadarConfig, range_profiles  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 150})

CHAINS = [["head", "chest", "pelvis"],
          ["pelvis", "thigh_L", "shank_L", "foot_L"], ["pelvis", "thigh_R", "shank_R", "foot_R"],
          ["chest", "upperarm_L", "forearm_L", "hand_L"], ["chest", "upperarm_R", "forearm_R", "hand_R"]]


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = RadarConfig()
    subj = Subject(subject_id=0, height=1.78, cadence_factor=1.0, arm_swing=1.0, rcs_scale=1.0, asymmetry=0.03)
    # Pick a seed that gives an approaching walk at a moderate angle.
    for seed in range(200):
        rng = np.random.default_rng(seed)
        rec = make_recording(subj, "nominal", rng, cfg, speed=1.3, keep_beat=True)
        m = rec.meta
        if m["approaching"] and 10 < abs(m["alpha_deg"]) < 20 and m["snr_db"] > 3:
            break

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)

    # (a) body model
    ax = fig.add_subplot(gs[0, 0])
    g = gait_params(subj, 1.3)
    t = np.linspace(0, g.stride_time, 7)[:-1]
    pos = body_frame_positions(subj, g, t)
    idx = {n: i for i, n in enumerate(SCATTERER_NAMES)}
    for k in range(len(t)):
        off = k * 0.75
        for ch in CHAINS:
            p = pos[k, [idx[n] for n in ch]]
            ax.plot(p[:, 0] + off, p[:, 2], "-", color="#7f8c9a", lw=1)
        ax.scatter(pos[k, :, 0] + off, pos[k, :, 2], s=40 * SCATTERER_WEIGHTS, color="#1f4e79", zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_ylabel("height (m)")
    ax.set_title("(a) 15-point scatterer body model\n(one stride, 6 snapshots)", fontsize=9)

    # (b) range-time map
    ax = fig.add_subplot(gs[0, 1])
    prof, rax = range_profiles(cfg, rec.beats[0])
    prof = prof - prof.mean(0, keepdims=True)
    keep = rax <= 14
    rt = 20 * np.log10(np.abs(prof[:, keep]).T + 1e-9)
    rt -= rt.max()
    ax.imshow(rt, aspect="auto", origin="lower", cmap="viridis", vmin=-40, vmax=0,
              extent=[0, cfg.duration, rax[keep][0], rax[keep][-1]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("range (m)")
    ax.set_title("(b) Range-time map after MTI\n(monostatic, walker approaching)", fontsize=9)

    # (c) monostatic micro-Doppler + torso ridge
    fd = cfg.doppler_axis
    frames_t = (np.arange(rec.spec_mono.shape[1]) * cfg.hop + cfg.win / 2) / cfg.prf
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(rec.spec_mono, aspect="auto", origin="lower", cmap="jet",
              extent=[frames_t[0], frames_t[-1], fd[0], fd[-1]])
    ridge = torso_doppler(rec.spec_mono, cfg)
    ax.plot(frames_t, ridge, "w.", ms=1.5, alpha=0.8)
    v_est = estimate_speed(rec.spec_mono, cfg)
    ax.set_ylim(-150, 330)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"(c) Monostatic micro-Doppler, torso ridge (white)\n"
                 f"true speed {m['speed']:.2f} m/s, radial estimate {v_est:.2f} m/s", fontsize=9)

    # (d) bistatic micro-Doppler
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rec.spec_bi, aspect="auto", origin="lower", cmap="jet",
              extent=[frames_t[0], frames_t[-1], fd[0], fd[-1]])
    ax.set_ylim(-150, 330)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Doppler (Hz)")
    ax.set_title("(d) Same walk seen by the bistatic receiver\n(different projection of the motion)", fontsize=9)

    # (e) cadence-velocity diagram
    ax = fig.add_subplot(gs[1, 1])
    freqs, cvd = cadence_velocity_diagram(rec.spec_mono, cfg)
    fk = freqs <= 6
    ax.imshow(cvd[:, fk], aspect="auto", origin="lower", cmap="magma",
              extent=[0, freqs[fk][-1], fd[0], fd[-1]])
    ax.set_ylim(-150, 330)
    ax.set_xlabel("cadence frequency (Hz)"); ax.set_ylabel("Doppler (Hz)")
    ax.set_title("(e) Cadence-velocity diagram\n(FFT along time of each Doppler bin)", fontsize=9)

    # (f) summed CVD profile
    ax = fig.add_subplot(gs[1, 2])
    prof_c = cvd[np.abs(fd) >= CLUTTER_NOTCH_HZ].sum(0)
    ax.plot(freqs[fk], prof_c[fk] / prof_c[fk].max(), color="#1f4e79")
    c_est = estimate_cadence(rec.spec_mono, cfg)
    ax.axvline(m["cadence"] / 60, color="k", ls="--", lw=1, label=f"true step rate ({m['cadence']:.1f} steps/min)")
    ax.axvline(c_est / 60, color="#e07b39", lw=1.2, label=f"estimate ({c_est:.1f} steps/min)")
    ax.axvspan(1.0, 2.6, color="#dfe8f1", alpha=0.6, lw=0, label="search band")
    ax.set_xlabel("cadence frequency (Hz)"); ax.set_ylabel("normalised magnitude")
    ax.set_title("(f) Summed CVD: peak = step frequency", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    fig.savefig(root / "figures/fig1_signal_chain.png", bbox_inches="tight")
    print("saved figures/fig1_signal_chain.png", m)


if __name__ == "__main__":
    main()
