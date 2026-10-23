"""Point-scatterer model of a walking human.

The model is a simplified, Boulic-inspired gait model:

* Stride duration follows the Boulic-Magnenat-Thalmann relation
  ``T_c = 1.346 / sqrt(RV)`` with ``RV = v / H_leg`` (relative velocity),
  scaled by a per-subject cadence factor, so cadence is *correlated* with
  speed and body height but not fully determined by them.
* Segment lengths follow Winter's anthropometric ratios.
* Hip amplitude is chosen so the stance-phase foot excursion matches the
  step length, and pelvis height is set by the longer (stance) leg, which
  produces the double vertical bounce per stride automatically.
* Each body segment is represented by one point scatterer at its midpoint,
  with an amplitude weight that roughly follows its radar cross-section.

The model is intentionally simple. It reproduces the main structures of a
walking micro-Doppler signature (torso line, limb swings, step periodicity),
not a full electromagnetic body model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Scatterer names and amplitude weights (approx. sqrt of relative RCS).
SCATTERERS = [
    ("head", 0.45),
    ("chest", 1.00),
    ("pelvis", 0.85),
    ("thigh_L", 0.55), ("thigh_R", 0.55),
    ("shank_L", 0.40), ("shank_R", 0.40),
    ("foot_L", 0.25), ("foot_R", 0.25),
    ("upperarm_L", 0.35), ("upperarm_R", 0.35),
    ("forearm_L", 0.25), ("forearm_R", 0.25),
    ("hand_L", 0.12), ("hand_R", 0.12),
]
SCATTERER_NAMES = [n for n, _ in SCATTERERS]
SCATTERER_WEIGHTS = np.array([w for _, w in SCATTERERS])


@dataclass
class Subject:
    """Anthropometry and individual gait style of one (synthetic) person."""

    subject_id: int
    height: float          # body height [m]
    cadence_factor: float  # individual multiplier on the Boulic cadence
    arm_swing: float       # multiplier on arm-swing amplitude
    rcs_scale: float       # overall reflectivity multiplier
    asymmetry: float       # left/right difference in hip amplitude (fraction)

    def to_dict(self) -> dict:
        return asdict(self)


def sample_subject(rng: np.random.Generator, subject_id: int) -> Subject:
    return Subject(
        subject_id=subject_id,
        height=float(rng.uniform(1.50, 1.95)),
        cadence_factor=float(np.clip(rng.normal(1.0, 0.06), 0.88, 1.12)),
        arm_swing=float(rng.uniform(0.6, 1.3)),
        rcs_scale=float(rng.uniform(0.7, 1.3)),
        asymmetry=float(rng.uniform(0.0, 0.08)),
    )


@dataclass
class GaitParams:
    speed: float           # walking speed [m/s]
    stride_time: float     # duration of one full gait cycle [s]
    cadence: float         # steps per minute
    step_length: float     # [m]


def gait_params(subject: Subject, speed: float, cadence: float | None = None) -> GaitParams:
    """Gait timing for a walking speed.

    By default cadence follows the Boulic relation (coupled to speed and body
    height). Passing ``cadence`` [steps/min] overrides it, which is used to
    generate data in which speed and cadence vary independently.
    """
    if cadence is not None:
        stride_time = 120.0 / cadence
        return GaitParams(speed=speed, stride_time=float(stride_time), cadence=float(cadence),
                          step_length=float(speed * stride_time / 2.0))
    leg = 0.53 * subject.height                 # hip height ~ leg length
    rv = speed / leg                            # relative velocity
    stride_time = 1.346 / np.sqrt(rv) / subject.cadence_factor
    return GaitParams(
        speed=speed,
        stride_time=float(stride_time),
        cadence=float(120.0 / stride_time),     # two steps per stride
        step_length=float(speed * stride_time / 2.0),
    )


def _pgauss(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    """Periodic Gaussian bump on the unit gait cycle."""
    d = (phase - center + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / width) ** 2)


def _leg_angles(phase, hip_amp_deg, speed_scale):
    """Hip flexion, knee flexion and ankle angles [rad] over the gait phase.

    phase = 0 is heel strike. Positive hip angle = thigh swung forward.
    """
    hip = np.deg2rad(8.0 + hip_amp_deg * np.cos(2 * np.pi * phase))
    knee = np.deg2rad(
        3.0
        + (6.0 + 10.0 * speed_scale) * _pgauss(phase, 0.13, 0.06)   # loading response
        + (40.0 + 15.0 * speed_scale) * _pgauss(phase, 0.70, 0.10)  # swing flexion
    )
    ankle = np.deg2rad(8.0 * _pgauss(phase, 0.40, 0.12) - 18.0 * _pgauss(phase, 0.62, 0.06))
    return hip, knee, ankle


def body_frame_positions(subject: Subject, gait: GaitParams, t: np.ndarray,
                         phase0: float = 0.0, phase: np.ndarray | None = None) -> np.ndarray:
    """Scatterer positions in a body frame that translates with the walker.

    Frame: x forward, y left, z up. The origin moves with the pelvis along x
    (pelvis x = 0) and sits on the ground (z = 0).

    ``phase`` (gait cycles, optional) overrides the constant-cadence phase
    ``t / stride_time + phase0`` so stride-to-stride variability can be added.

    Returns an array of shape (len(t), n_scatterers, 3).
    """
    H = subject.height
    L_thigh, L_shank, L_foot = 0.245 * H, 0.246 * H, 0.152 * H
    ankle_h = 0.039 * H
    L_uarm, L_farm, L_hand = 0.186 * H, 0.146 * H, 0.108 * H
    hip_y, sh_y = 0.09 * H, 0.13 * H

    speed_scale = float(np.clip(gait.speed / 1.3, 0.4, 1.4))
    # Hip amplitude so that stance foot excursion ~ step length.
    leg = L_thigh + L_shank
    sin_a = np.clip(gait.step_length / (2.0 * leg * np.cos(np.deg2rad(8.0))), 0.05, 0.53)
    hip_amp = np.rad2deg(np.arcsin(sin_a))

    if phase is None:
        phase = t / gait.stride_time + phase0
    out = {}
    leg_vertical = []
    leg_pts = {}
    for side, offset, sign_y, amp_mult in (
        ("L", 0.0, +1, 1.0 + subject.asymmetry / 2),
        ("R", 0.5, -1, 1.0 - subject.asymmetry / 2),
    ):
        ph = phase + offset
        hip, knee, ankle = _leg_angles(ph, hip_amp * amp_mult, speed_scale)
        shank_abs = hip - knee
        # Positions relative to the hip joint (sagittal plane x, z).
        knee_x, knee_z = L_thigh * np.sin(hip), -L_thigh * np.cos(hip)
        ank_x = knee_x + L_shank * np.sin(shank_abs)
        ank_z = knee_z - L_shank * np.cos(shank_abs)
        foot_ang = shank_abs + ankle
        foot_mid_x = ank_x + 0.5 * L_foot * np.cos(foot_ang) - 0.1 * L_foot
        foot_mid_z = ank_z + 0.5 * L_foot * np.sin(foot_ang)
        leg_vertical.append(-ank_z)
        leg_pts[side] = dict(
            y=sign_y * hip_y,
            thigh=(0.5 * knee_x, 0.5 * knee_z),
            shank=(0.5 * (knee_x + ank_x), 0.5 * (knee_z + ank_z)),
            foot=(foot_mid_x, foot_mid_z),
        )

    # Pelvis height set by the longer (stance) leg -> natural vertical bounce.
    hip_z = ankle_h + np.maximum(leg_vertical[0], leg_vertical[1])
    sway = 0.02 * np.sin(2 * np.pi * phase)          # lateral sway, once per stride
    lean = np.deg2rad(3.0 + 2.0 * speed_scale)       # forward trunk lean

    def P(x, y, z):
        return np.stack([x, y + sway, z], axis=-1)

    zeros = np.zeros_like(t)
    for side in ("L", "R"):
        lp = leg_pts[side]
        y = lp["y"] + zeros
        out[f"thigh_{side}"] = P(lp["thigh"][0], y, hip_z + lp["thigh"][1])
        out[f"shank_{side}"] = P(lp["shank"][0], y, hip_z + lp["shank"][1])
        out[f"foot_{side}"] = P(lp["foot"][0], y, hip_z + lp["foot"][1])

    # Trunk and head.
    out["pelvis"] = P(zeros, zeros, hip_z + 0.03 * H)
    out["chest"] = P(0.19 * H * np.sin(lean) + zeros, zeros, hip_z + 0.19 * H * np.cos(lean))
    out["head"] = P(0.40 * H * np.sin(lean) + zeros, zeros, hip_z + 0.40 * H * np.cos(lean))

    # Arms swing opposite to the ipsilateral leg.
    arm_amp = np.deg2rad(subject.arm_swing * (8.0 + 14.0 * speed_scale))
    sh_z = hip_z + 0.29 * H
    sh_x = 0.29 * H * np.sin(lean)
    for side, offset, sign_y in (("L", 0.5, +1), ("R", 0.0, -1)):
        ph = phase + offset
        shoulder = np.deg2rad(3.0) + arm_amp * np.cos(2 * np.pi * ph)
        elbow = np.deg2rad(15.0) + 0.6 * arm_amp * 0.5 * (1 + np.cos(2 * np.pi * ph))
        fore = shoulder + elbow
        el_x, el_z = L_uarm * np.sin(shoulder), -L_uarm * np.cos(shoulder)
        wr_x, wr_z = el_x + L_farm * np.sin(fore), el_z - L_farm * np.cos(fore)
        hd_x, hd_z = wr_x + 0.5 * L_hand * np.sin(fore), wr_z - 0.5 * L_hand * np.cos(fore)
        y = sign_y * sh_y + zeros
        out[f"upperarm_{side}"] = P(sh_x + 0.5 * el_x, y, sh_z + 0.5 * el_z)
        out[f"forearm_{side}"] = P(sh_x + 0.5 * (el_x + wr_x), y, sh_z + 0.5 * (el_z + wr_z))
        out[f"hand_{side}"] = P(sh_x + hd_x, y, sh_z + hd_z)

    return np.stack([out[n] for n in SCATTERER_NAMES], axis=1)


def world_positions(body_pos: np.ndarray, t: np.ndarray, start_xy, heading_rad: float,
                    speed: float, distance: np.ndarray | None = None) -> np.ndarray:
    """Place the body-frame trajectory in the world.

    The walker starts at ``start_xy`` on the ground and walks in a straight
    line with direction ``heading_rad`` (measured from the +x axis).
    ``distance`` (optional, metres travelled at each t) overrides ``speed * t``.
    """
    if distance is None:
        distance = speed * t
    d = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    left = np.array([-d[1], d[0]])
    R = np.array([[d[0], left[0], 0.0], [d[1], left[1], 0.0], [0.0, 0.0, 1.0]])
    world = body_pos @ R.T
    world[..., 0] += start_xy[0] + distance[:, None] * d[0]
    world[..., 1] += start_xy[1] + distance[:, None] * d[1]
    return world


def smooth_variation(rng: np.random.Generator, t: np.ndarray, rel_std: float,
                     corr_time: float = 1.0) -> np.ndarray:
    """Zero-mean, smoothly varying relative fluctuation (e.g. +-3 %)."""
    dt = t[1] - t[0]
    n_knots = int(np.ceil(t[-1] / corr_time)) + 2
    knots_t = np.arange(n_knots) * corr_time
    vals = rng.normal(0.0, rel_std, n_knots)
    x = np.interp(t, knots_t, vals)
    k = int(max(1, corr_time / dt / 2))
    x = np.convolve(x, np.ones(k) / k, mode="same")
    return x - x.mean()
