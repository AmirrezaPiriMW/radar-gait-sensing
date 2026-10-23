"""Real-data path: Infineon BGT60 (60 GHz FMCW) recordings -> micro-Doppler -> gait estimates.

The BGT60 records in *frames*: a burst of ``n_chirps`` chirps (chirp
repetition time ``t_chirp``), then an idle gap until the next frame
(frame repetition time ``t_frame``). Slow time is therefore not continuous,
so instead of an STFT over a continuous signal, each frame gives one Doppler
spectrum (FFT across its chirps) and consecutive frames form the spectrogram:

    raw (frames, rx, chirps, samples)
      -> range FFT per chirp -> MTI (mean over the whole recording per range bin)
      -> range gate -> Doppler FFT across the chirps of each frame
      -> power summed over range bins and RX channels -> dB, normalised

The resulting spectrogram (Doppler x frames) has the same layout as the
simulated one, so the physics baseline (torso Doppler, cadence-velocity
diagram) runs unchanged through ``BGT60Config``, which exposes the same
attributes as ``RadarConfig``.

Two physical checks matter at 60 GHz (wavelength ~5 mm):

* Doppler ambiguity: +-1 / (2 t_chirp), i.e. radial speeds up to
  lambda / (4 t_chirp). The swinging foot reaches ~4.4x walking speed, so
  covering the feet needs t_chirp <= ~225 us at 1.3 m/s and <= ~160 us at
  1.8 m/s. With longer chirp times the torso line is still usable for speed,
  but limb micro-Doppler folds over.
* Range: with real-valued IF sampling the maximum range is
  fs / 2 * c / (2 * slope). A 5.5 GHz sweep in 64 us reaches only ~1.7 m at
  2 MHz sampling, so walking recordings need a smaller bandwidth (e.g. 1 GHz)
  or more samples per chirp.
* Cadence: the frame rate must be well above twice the step frequency
  (step rate <= 2.6 Hz); 20 Hz or more is comfortable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

C0 = 299_792_458.0


@dataclass
class BGT60Config:
    # Defaults: a walking-friendly set-up (9.6 m range, feet unaliased up to 1.8 m/s).
    # Real recordings override these from their config file.
    f_start: float = 60e9              # Hz
    f_end: float = 61e9                # Hz
    n_samples: int = 128               # samples per chirp
    n_chirps: int = 64                 # chirps per frame
    sample_rate: float = 2e6           # ADC sample rate [Hz]
    t_chirp: float = 150e-6            # chirp repetition time [s]
    t_frame: float = 0.04              # frame repetition time [s]
    range_gate: tuple = (0.5, 8.0)     # processed range [m]
    dyn_range_db: float = 45.0
    doppler_pad: int = 2               # zero-padding factor of the Doppler FFT

    # --- attributes shared with RadarConfig (used by gaitradar.baseline) ---
    @property
    def fc(self) -> float:
        return 0.5 * (self.f_start + self.f_end)

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def ramp_time(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def slope(self) -> float:
        return self.bandwidth / self.ramp_time

    @property
    def range_resolution(self) -> float:
        return C0 / (2 * self.bandwidth)

    @property
    def frame_rate(self) -> float:
        return 1.0 / self.t_frame

    @property
    def n_doppler(self) -> int:
        return self.n_chirps * self.doppler_pad

    @property
    def doppler_axis(self) -> np.ndarray:
        return np.fft.fftshift(np.fft.fftfreq(self.n_doppler, d=self.t_chirp))

    @property
    def max_range(self) -> float:
        """Largest range [m] before the real-valued IF exceeds the ADC Nyquist frequency."""
        return 0.5 * self.sample_rate * C0 / (2 * self.slope)

    @property
    def max_unambiguous_speed(self) -> float:
        """Largest radial speed [m/s] without Doppler aliasing (monostatic)."""
        return self.wavelength / (4 * self.t_chirp)

    def check(self, max_walking_speed: float = 1.8) -> list[str]:
        """Human-readable warnings about the recording set-up."""
        w = []
        foot = 4.4 * max_walking_speed            # swinging foot ~ 4.4x walking speed
        if self.max_unambiguous_speed < 1.2 * max_walking_speed:
            w.append(f"Torso Doppler aliases above {self.max_unambiguous_speed:.2f} m/s: speed estimates are unreliable.")
        elif self.max_unambiguous_speed < foot:
            w.append(f"Limb Doppler aliases above {self.max_unambiguous_speed:.2f} m/s (feet reach ~{foot:.1f} m/s): "
                     "torso speed is fine, but limb features fold over.")
        if self.max_range < self.range_gate[1]:
            w.append(f"Maximum range is {self.max_range:.1f} m (ADC Nyquist) but the range gate extends to "
                     f"{self.range_gate[1]:.1f} m: targets beyond fold over; reduce bandwidth or the range gate.")
        if self.ramp_time > self.t_chirp:
            w.append("Chirp ramp (samples / sample rate) is longer than the chirp repetition time.")
        if self.frame_rate < 10:
            w.append(f"Frame rate {self.frame_rate:.1f} Hz is low for cadence (step rate up to 2.6 Hz).")
        if self.n_chirps * self.t_chirp > self.t_frame:
            w.append("Chirp burst is longer than the frame period: check t_chirp / t_frame.")
        return w


# Key names used by Infineon tools (Radar Fusion GUI / Radar SDK "fmcw_single_shape").
_KEYS = {
    "f_start": ["start_frequency_Hz", "lower_frequency_Hz"],
    "f_end": ["end_frequency_Hz", "upper_frequency_Hz"],
    "n_samples": ["num_samples_per_chirp", "num_samples"],
    "n_chirps": ["num_chirps_per_frame", "num_chirps"],
    "sample_rate": ["sample_rate_Hz", "adc_sample_rate_Hz"],
    "t_chirp": ["chirp_repetition_time_s"],
    "t_frame": ["frame_repetition_time_s"],
}


def _find(d, names):
    """Depth-first search for the first key in ``names`` inside nested dicts/lists."""
    if isinstance(d, dict):
        for n in names:
            if n in d:
                return d[n]
        for v in d.values():
            r = _find(v, names)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, names)
            if r is not None:
                return r
    return None


def config_from_json(path: str | Path, **overrides) -> BGT60Config:
    """Read a recording's JSON config. Missing keys fall back to defaults and are
    reported; any field can be overridden explicitly (e.g. t_chirp=200e-6)."""
    cfg_json = json.loads(Path(path).read_text())
    kw, missing = {}, []
    for field, names in _KEYS.items():
        v = _find(cfg_json, names)
        if v is None:
            missing.append(field)
        else:
            kw[field] = type(getattr(BGT60Config, field))(v)
    kw.update(overrides)
    cfg = BGT60Config(**kw)
    cfg.missing_keys = [m for m in missing if m not in overrides]
    return cfg


def load_raw(path: str | Path) -> np.ndarray:
    """Load raw ADC data as (frames, rx, chirps, samples), float32, DC removed per chirp."""
    x = np.load(path)
    if x.ndim == 3:                       # (frames, chirps, samples), single RX
        x = x[:, None]
    if x.ndim != 4:
        raise ValueError(f"expected (frames, rx, chirps, samples), got shape {x.shape}")
    x = x.astype(np.float32)
    return x - x.mean(axis=-1, keepdims=True)


def micro_doppler_frames(cfg: BGT60Config, raw: np.ndarray) -> np.ndarray:
    """Raw frames -> normalised micro-Doppler spectrogram (n_doppler, n_frames).

    The BGT60 delivers real-valued IF samples, so the range FFT keeps the
    positive-frequency half. Positive Doppler = approaching (path shrinking).
    """
    n_f, n_rx, n_c, n_s = raw.shape
    win_r = np.hanning(n_s).astype(np.float32)
    rp = np.fft.rfft(raw * win_r, axis=-1)                         # (F, RX, C, bins)
    rng_axis = np.arange(rp.shape[-1]) * cfg.sample_rate / n_s * C0 / (2 * cfg.slope)
    rp = rp - rp.mean(axis=(0, 2), keepdims=True)                  # MTI over the whole recording
    gate = (rng_axis >= cfg.range_gate[0]) & (rng_axis <= cfg.range_gate[1])
    rp = rp[..., gate]
    win_d = np.hanning(n_c).astype(np.float32)[None, None, :, None]
    dop = np.fft.fft(np.conj(rp) * win_d, n=cfg.n_doppler, axis=2)  # conj: approaching -> +f
    power = (np.abs(dop) ** 2).sum(axis=(1, 3))                    # sum RX and range bins -> (F, D)
    power = np.fft.fftshift(power, axes=1).T                       # (D, F)
    p_db = 10 * np.log10(power + 1e-12)
    p_db -= p_db.max()
    p_db = np.clip(p_db, -cfg.dyn_range_db, 0.0)
    return ((p_db + cfg.dyn_range_db) / cfg.dyn_range_db).astype(np.float32)


def simulate_bgt60_recording(cfg: BGT60Config, subject, speed: float, n_frames: int, rng: np.random.Generator,
                             start_xy=(5.5, 0.0), heading_rad: float = np.pi, snr_db: float = 10.0,
                             radar_xyz=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Simulate a BGT60-format recording (frames, 1, chirps, samples) of one walk.

    Used to test the real-data pipeline end to end, and as the starting point
    for a 60 GHz sim-to-real study. Real-valued IF signal, stop-and-hop.
    """
    from .kinematics import SCATTERER_WEIGHTS, body_frame_positions, gait_params, world_positions
    t = (np.arange(n_frames)[:, None] * cfg.t_frame + np.arange(cfg.n_chirps)[None, :] * cfg.t_chirp).ravel()
    gait = gait_params(subject, speed)
    pos = world_positions(body_frame_positions(subject, gait, t), t, start_xy, heading_rad, speed)
    radar = np.asarray(radar_xyz)
    r = np.linalg.norm(pos - radar, axis=-1)                         # (T, K)
    amp = SCATTERER_WEIGHTS[None, :] * 25.0 / np.maximum(r ** 2, 0.25)
    tau = 2 * r / C0
    n = np.arange(cfg.n_samples) / cfg.sample_rate
    phase = 2 * np.pi * (cfg.slope * tau[..., None] * n + cfg.f_start * tau[..., None])
    sig = np.einsum("tk,tkn->tn", amp.astype(np.float32), np.cos(phase).astype(np.float32))
    sig += rng.standard_normal(sig.shape).astype(np.float32) * 10 ** (-snr_db / 20)
    return sig.reshape(n_frames, cfg.n_chirps, cfg.n_samples)[:, None]
