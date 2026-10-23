"""Scenario sampling and dataset generation.

A *deployment* has one monostatic FMCW radar (Tx/Rx co-located at the
origin) and one additional bistatic receiver node. Each recording is a
4 s walk on a straight line with random speed, direction, start point,
SNR and static clutter. Both receive channels are simulated, so the same
recording can be used for monostatic-only or multistatic processing.

Conditions
----------
``alpha`` is the *heading angle* between the walking direction and the
radar's boresight axis (the x axis). Because walkers start up to 1 m off
axis and pass at different distances, the angle to the actual line of
sight (``los_aspect_deg``) differs from alpha and is stored separately.

``nominal``  : |alpha| <= 40 deg, SNR in [-6, 8] dB
``oblique``  : 45 deg <= |alpha| <= 70 deg (heading range unseen in training)
``low_snr``  : nominal geometry, SNR in [-22, -14] dB (SNR range unseen in training)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import (SCATTERER_WEIGHTS, Subject, body_frame_positions, gait_params,
                         sample_subject, smooth_variation, world_positions)
from .radar import RadarConfig, micro_doppler, simulate_beat_signal

TX = np.array([0.0, 0.0, 1.1])           # monostatic radar (Tx and Rx 1)
RX_BISTATIC = np.array([6.0, 7.5, 1.2])  # separate receiver node (Rx 2)

# Decoupled sampling: speed and cadence drawn jointly uniform, accepted only if
# the resulting step length is plausible for the person's height.
DECOUPLED_SPEED = (0.5, 1.8)          # m/s
DECOUPLED_CADENCE = (62.0, 150.0)     # steps/min (inside the baseline's CVD band)
DECOUPLED_STEP_REL = (0.22, 0.50)     # step length / body height


def sample_decoupled_gait(subject: Subject, rng: np.random.Generator) -> tuple[float, float]:
    """Draw (speed, cadence) independently, constrained to plausible step lengths."""
    while True:
        v = rng.uniform(*DECOUPLED_SPEED)
        c = rng.uniform(*DECOUPLED_CADENCE)
        rel = 60.0 * v / c / subject.height
        if DECOUPLED_STEP_REL[0] <= rel <= DECOUPLED_STEP_REL[1]:
            return float(v), float(c)


CONDITIONS = {
    "nominal": dict(aspect=(0.0, 40.0), snr=(-6.0, 8.0)),
    "oblique": dict(aspect=(45.0, 70.0), snr=(-6.0, 8.0)),
    "low_snr": dict(aspect=(0.0, 40.0), snr=(-22.0, -14.0)),
}


@dataclass
class Recording:
    spec_mono: np.ndarray
    spec_bi: np.ndarray
    meta: dict
    beats: list | None = None       # raw beat signals (only if keep_beat=True)
    positions: np.ndarray | None = None


def make_recording(subject: Subject, condition: str, rng: np.random.Generator,
                   cfg: RadarConfig | None = None, speed: float | None = None,
                   keep_beat: bool = False, decoupled: bool = False) -> Recording:
    """Simulate one walk seen by the monostatic radar and the bistatic receiver.

    ``decoupled=True`` draws speed and cadence independently (within plausible
    step lengths) instead of using the speed-coupled Boulic cadence.
    """
    cfg = cfg or RadarConfig()
    cond = CONDITIONS[condition]
    t = cfg.slow_time
    T = cfg.duration

    if decoupled:
        v, cad = sample_decoupled_gait(subject, rng)
        gait = gait_params(subject, v, cadence=cad)
    else:
        v = float(rng.uniform(0.5, 1.8)) if speed is None else float(speed)
        gait = gait_params(subject, v)

    alpha = np.deg2rad(rng.uniform(*cond["aspect"])) * rng.choice([-1.0, 1.0])
    approaching = bool(rng.random() < 0.5)
    if approaching:
        heading = np.pi + alpha
        x0 = 1.5 + v * T * np.cos(alpha) + rng.uniform(0.5, 2.5)
    else:
        heading = alpha
        x0 = rng.uniform(1.5, 2.5)
    start = np.array([x0, rng.uniform(-1.0, 1.0)])

    # Stride-to-stride variability: +-3 % in stride frequency and speed.
    dt = t[1] - t[0]
    f_inst = (1.0 / gait.stride_time) * (1.0 + smooth_variation(rng, t, 0.03))
    phase = float(rng.random()) + np.cumsum(f_inst) * dt
    v_inst = v * (1.0 + smooth_variation(rng, t, 0.03))
    distance = np.cumsum(v_inst) * dt
    body = body_frame_positions(subject, gait, t, phase=phase)
    pos = world_positions(body, t, start, heading, v, distance=distance)

    # Static clutter (walls, furniture) with tiny vibration (residual near 0 Hz).
    n_cl = 5
    cl_base = np.column_stack([rng.uniform(1, 15, n_cl), rng.uniform(-6, 8, n_cl),
                               rng.uniform(0, 2.5, n_cl)])
    vib = 1e-4 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5, n_cl)[None, :] * t[:, None])
    cl_pos = cl_base[None] + vib[..., None] * np.array([1.0, 0.0, 0.0])
    cl_amp = rng.uniform(1.0, 15.0, n_cl)

    snr = float(rng.uniform(*cond["snr"]))
    radar_h = float(rng.uniform(0.9, 1.3))
    tx = TX.copy(); tx[2] = radar_h
    w = SCATTERER_WEIGHTS * subject.rcs_scale

    specs, beats = [], []
    for rx in (tx, RX_BISTATIC):
        pos_all = np.concatenate([pos, cl_pos], axis=1)
        w_all = np.concatenate([w, cl_amp])
        beat = simulate_beat_signal(cfg, pos_all, w_all, tx, rx, snr, rng=rng)
        specs.append(micro_doppler(cfg, beat))
        if keep_beat:
            beats.append(beat)

    # Aspect relative to the radar line of sight at mid-recording.
    mid = pos[len(t) // 2, 1, :2]            # chest
    los = (tx[:2] - mid) / np.linalg.norm(tx[:2] - mid)
    vel = np.array([np.cos(heading), np.sin(heading)])
    los_aspect = float(np.rad2deg(np.arccos(np.clip(abs(los @ vel), 0, 1))))

    meta = dict(
        subject_id=subject.subject_id, height=subject.height,
        speed=float(v_inst.mean()), cadence=float(120.0 * f_inst.mean()),
        step_length=float(v_inst.mean() / (2.0 * f_inst.mean())),
        stride_time=gait.stride_time, condition=condition,
        alpha_deg=float(np.rad2deg(alpha)), los_aspect_deg=los_aspect,
        approaching=approaching, snr_db=snr, radar_height=radar_h,
        start_x=float(start[0]), start_y=float(start[1]),
    )
    if decoupled:
        meta["sampling"] = "decoupled"
    return Recording(specs[0], specs[1], meta,
                     beats=beats if keep_beat else None, positions=pos if keep_beat else None)


def make_subjects(n: int, seed: int) -> list[Subject]:
    rng = np.random.default_rng(seed)
    return [sample_subject(rng, i) for i in range(n)]
