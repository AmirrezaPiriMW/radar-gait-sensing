"""Classical signal-processing baseline for gait speed and cadence.

* **Speed** from the torso Doppler line: per frame, the strongest return
  outside the clutter notch (the torso dominates the RCS) is located and
  refined to sub-bin precision with a power-weighted centroid over the peak's
  neighbouring bins. The median over time gives the torso Doppler ``f_t``,
  converted with ``v = f_t * lambda / 2`` (monostatic). A single monostatic
  radar only sees the *radial* component, so this estimate is biased low
  whenever the walker does not move along the line of sight.
* **Cadence** from the cadence-velocity diagram (CVD): an FFT along time of
  every Doppler bin of the spectrogram. Limb micro-Doppler repeats once per
  step, so the summed CVD peaks at the step frequency. The search band
  1.0-2.6 Hz (60-156 steps/min) covers every walk in the evaluation splits.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend

from .radar import RadarConfig

STEP_FREQ_BAND = (1.0, 2.6)   # Hz, i.e. 60-156 steps/min
CLUTTER_NOTCH_HZ = 8.0
CENTROID_HALF_WIDTH = 2       # bins on each side of the peak used for sub-bin refinement


def _linear_power(spec: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    return 10 ** ((spec * cfg.dyn_range_db - cfg.dyn_range_db) / 10.0)


def torso_doppler(spec: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Frame-wise torso Doppler [Hz]: spectral peak outside the clutter notch,
    refined with a power-weighted centroid over +-2 bins."""
    fd = cfg.doppler_axis
    p = _linear_power(spec, cfg).copy()
    p[np.abs(fd) < CLUTTER_NOTCH_HZ] = 0.0
    k = np.array([0.25, 0.5, 0.25])                       # light smoothing against speckle
    ps = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 0, p)
    peak = ps.argmax(axis=0)
    out = np.empty(p.shape[1])
    for j, i in enumerate(peak):
        lo, hi = max(0, i - CENTROID_HALF_WIDTH), min(len(fd), i + CENTROID_HALF_WIDTH + 1)
        w = ps[lo:hi, j]
        out[j] = np.sum(w * fd[lo:hi]) / np.sum(w)
    return out


def estimate_speed(spec: np.ndarray, cfg: RadarConfig) -> float:
    """Monostatic radial-speed estimate [m/s] from the torso Doppler line."""
    ft = np.median(torso_doppler(spec, cfg))
    return float(abs(ft) * cfg.wavelength / 2.0)


def cadence_velocity_diagram(spec: np.ndarray, cfg: RadarConfig, n_fft: int = 4096):
    """Return (cadence frequencies [Hz], CVD magnitude (n_doppler, n_freq))."""
    x = detrend(spec, axis=1, type="linear")
    x = x * np.hanning(x.shape[1])[None, :]
    cvd = np.abs(np.fft.rfft(x, n=n_fft, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / cfg.frame_rate)
    return freqs, cvd


def estimate_cadence(spec: np.ndarray, cfg: RadarConfig) -> float:
    """Cadence in steps/min from the summed cadence-velocity diagram."""
    freqs, cvd = cadence_velocity_diagram(spec, cfg)
    fd = cfg.doppler_axis
    profile = cvd[np.abs(fd) >= CLUTTER_NOTCH_HZ].sum(axis=0)
    band = (freqs >= STEP_FREQ_BAND[0]) & (freqs <= STEP_FREQ_BAND[1])
    idx = np.flatnonzero(band)
    i = idx[np.argmax(profile[band])]
    # parabolic interpolation of the peak
    a, b, c = profile[i - 1], profile[i], profile[i + 1]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    f_step = freqs[i] + delta * (freqs[1] - freqs[0])
    return float(60.0 * f_step)
