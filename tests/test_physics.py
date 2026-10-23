"""Sanity checks of the kinematic model, radar simulator and baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitradar import uq  # noqa: E402
from gaitradar.baseline import estimate_cadence, estimate_speed, torso_doppler  # noqa: E402
from gaitradar.dataset import TX, make_recording  # noqa: E402
from gaitradar.kinematics import (SCATTERER_NAMES, Subject, body_frame_positions,  # noqa: E402
                                  gait_params, world_positions)
from gaitradar.radar import RadarConfig, micro_doppler, range_profiles, simulate_beat_signal  # noqa: E402

SUBJ = Subject(subject_id=0, height=1.78, cadence_factor=1.0, arm_swing=1.0, rcs_scale=1.0, asymmetry=0.0)


def test_cadence_is_realistic():
    # Normal walking (~1.3 m/s) should give ~100-120 steps/min.
    g = gait_params(SUBJ, 1.3)
    assert 95 < g.cadence < 125
    assert g.step_length == pytest.approx(1.3 * 60 / g.cadence, rel=1e-6)
    # Faster walking -> higher cadence and longer steps.
    g2 = gait_params(SUBJ, 1.7)
    assert g2.cadence > g.cadence and g2.step_length > g.step_length


def test_limbs_average_to_body_speed():
    # Every body point is periodic in the body frame, so over whole strides its
    # mean forward velocity equals the walking speed.
    cfg = RadarConfig()
    g = gait_params(SUBJ, 1.2)
    t = np.arange(0, 3 * g.stride_time, cfg.chirp_time)
    pos = world_positions(body_frame_positions(SUBJ, g, t), t, (0.0, 0.0), 0.0, 1.2)
    mean_v = (pos[-1, :, 0] - pos[0, :, 0]) / (t[-1] - t[0])
    assert np.allclose(mean_v, 1.2, atol=0.02)
    # Pelvis bounces twice per stride, by a few centimetres.
    z = pos[:, SCATTERER_NAMES.index("pelvis"), 2]
    assert 0.01 < z.max() - z.min() < 0.12


def test_point_target_range_and_doppler():
    # A single point moving straight towards a monostatic radar at 1 m/s from 6 m.
    cfg = RadarConfig(duration=1.0)
    t = cfg.slow_time
    p = np.zeros((len(t), 1, 3))
    p[:, 0, 0] = 6.0 - 1.0 * t
    p[:, 0, 2] = TX[2]
    beat = simulate_beat_signal(cfg, p, np.array([1.0]), TX, TX, snr_db=60)
    prof, rax = range_profiles(cfg, beat)
    assert abs(rax[np.abs(prof[0]).argmax()] - 6.0) <= 0.4          # within one range bin
    spec = micro_doppler(cfg, beat)
    fd = np.median(torso_doppler(spec, cfg))
    expected = 2 * 1.0 / cfg.wavelength                              # ~38.7 Hz, approaching -> positive
    assert abs(fd - expected) < 4.0


def test_baseline_on_clean_walk():
    cfg = RadarConfig()
    rng = np.random.default_rng(0)
    rec = make_recording(SUBJ, "nominal", rng, cfg, speed=1.3)
    m = rec.meta
    assert abs(estimate_cadence(rec.spec_mono, cfg) - m["cadence"]) < 5.0
    # Monostatic speed is the radial component, so it can only underestimate (plus noise).
    v = estimate_speed(rec.spec_mono, cfg)
    assert v < m["speed"] + 0.1


def test_uq_metrics_on_perfectly_calibrated_gaussian():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 2.0, 20000)
    y = rng.normal(0, sigma)
    mu = np.zeros_like(y)
    assert uq.coverage(y, mu, sigma, 0.9) == pytest.approx(0.9, abs=0.01)
    assert uq.calibration_error(y, mu, sigma) < 0.01
    assert uq.fit_sigma_scale(y, mu, sigma) == pytest.approx(1.0, abs=0.02)
    assert uq.fit_sigma_scale(y, mu, 2 * sigma) == pytest.approx(0.5, abs=0.01)


def test_doppler_window_covers_fastest_limb():
    # The swinging foot is the fastest point; at the top speed (1.8 m/s, +3 %
    # variability) its monostatic Doppler must stay inside the kept window.
    cfg = RadarConfig()
    worst = 0.0
    for h in (1.50, 1.95):
        s = Subject(subject_id=0, height=h, cadence_factor=1.12, arm_swing=1.3, rcs_scale=1.0, asymmetry=0.08)
        v = 1.8 * 1.03
        g = gait_params(s, v)
        t = np.arange(0, 2 * g.stride_time, 1e-3)
        vx = np.gradient(body_frame_positions(s, g, t)[..., 0], 1e-3, axis=0) + v
        worst = max(worst, 2 * vx.max() / cfg.wavelength)
    assert worst < cfg.doppler_axis[-1]


def test_doppler_flip_is_exact_mirror():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import doppler_flip
    cfg = RadarConfig()
    fd = cfg.doppler_axis
    x = torch.from_numpy(np.tile(fd[None, None, :, None], (1, 1, 1, 3)).astype(np.float32))
    y = doppler_flip(x)[0, 0, 1:, 0].numpy()
    assert np.allclose(y, -fd[1:])


def test_splits_are_subject_disjoint():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_dataset import SPLITS
    tr, va, te = (set(SPLITS[k][0]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert set(SPLITS["oblique"][0]) == te and set(SPLITS["low_snr"][0]) == te


def test_decoupled_sampling_weakens_speed_cadence_coupling():
    from gaitradar.dataset import DECOUPLED_STEP_REL, sample_decoupled_gait
    from gaitradar.kinematics import sample_subject
    rng = np.random.default_rng(0)
    subjects = [sample_subject(rng, i) for i in range(20)]
    v, c, rel = [], [], []
    for k in range(4000):
        s = subjects[k % 20]
        vi, ci = sample_decoupled_gait(s, rng)
        v.append(vi); c.append(ci); rel.append(60 * vi / ci / s.height)
        g = gait_params(s, vi, cadence=ci)
        assert g.cadence == pytest.approx(ci) and g.step_length == pytest.approx(60 * vi / ci)
    assert DECOUPLED_STEP_REL[0] - 1e-9 <= min(rel) and max(rel) <= DECOUPLED_STEP_REL[1] + 1e-9
    assert np.corrcoef(v, c)[0, 1] < 0.75    # coupled Boulic data: r ~ 0.9


def test_bgt60_pipeline_on_simulated_recording(tmp_path):
    from gaitradar.bgt60 import BGT60Config, micro_doppler_frames, simulate_bgt60_recording
    cfg = BGT60Config(f_start=60e9, f_end=61e9, t_chirp=150e-6, t_frame=0.04, n_chirps=64, n_samples=128, sample_rate=2e6)
    assert not cfg.check()                       # 150 us chirps cover limb Doppler up to 1.8 m/s
    raw = simulate_bgt60_recording(cfg, SUBJ, speed=1.2, n_frames=125, rng=np.random.default_rng(3),
                                   start_xy=(6.5, 0.0), heading_rad=np.pi, snr_db=15)
    assert raw.shape == (125, 1, 64, 128)
    spec = micro_doppler_frames(cfg, raw)
    g = gait_params(SUBJ, 1.2)
    assert abs(estimate_speed(spec, cfg) - 1.2) < 0.1          # walking straight at the radar
    assert abs(estimate_cadence(spec, cfg) - g.cadence) < 5.0
    assert any("aliases" in w for w in BGT60Config(f_start=60e9, f_end=61e9, t_chirp=500e-6).check())
    assert any("Maximum range" in w for w in BGT60Config(f_start=58e9, f_end=63.5e9).check())
