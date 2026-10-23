"""FMCW radar simulation (monostatic and bistatic) and micro-Doppler processing.

Signal model (stop-and-hop, one complex sample per fast-time instant):

    s[m, n] = sum_k a_k[m] * exp(j 2 pi ( S * L_k[m] / c * n / fs + L_k[m] / lambda ))

with ``L_k = |p_k - tx| + |p_k - rx|`` the propagation path length of
scatterer k at chirp m. Monostatic radar is the special case tx == rx.
Amplitudes follow the radar equation (``a ~ w / (R_tx * R_rx)``), static
clutter scatterers are added, and complex white noise sets the SNR.

The default parameters mimic a 5.8 GHz, 400 MHz bandwidth FMCW radar with a
1 ms chirp, similar to sensors widely used in radar human-activity research.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

C0 = 299_792_458.0


@dataclass
class RadarConfig:
    fc: float = 5.8e9            # carrier frequency [Hz]
    bandwidth: float = 400e6     # sweep bandwidth [Hz]
    chirp_time: float = 1e-3     # chirp duration = PRI [s]
    n_fast: int = 64             # fast-time samples per chirp
    duration: float = 4.0        # observation time [s]
    range_gate: tuple = (0.8, 16.0)  # processed (equivalent one-way) range [m]
    # STFT
    win: int = 128               # samples (128 ms) -> native Doppler resolution 7.8 Hz
    hop: int = 32                # samples (32 ms)
    nfft: int = 256              # 2x zero padding -> 3.9 Hz bin spacing
    # Doppler bins kept around 0 Hz: 192 bins = -375 ... +371 Hz. The fastest
    # body point (the swinging foot, ~4.4x walking speed, i.e. ~8.3 m/s at
    # 1.85 m/s) produces ~320 Hz, so the full signature fits inside the window.
    n_doppler: int = 192
    dyn_range_db: float = 45.0   # dynamic range kept after normalisation

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def slope(self) -> float:
        return self.bandwidth / self.chirp_time

    @property
    def prf(self) -> float:
        return 1.0 / self.chirp_time

    @property
    def n_chirps(self) -> int:
        return int(round(self.duration * self.prf))

    @property
    def slow_time(self) -> np.ndarray:
        return np.arange(self.n_chirps) * self.chirp_time

    @property
    def frame_rate(self) -> float:
        return self.prf / self.hop

    @property
    def doppler_axis(self) -> np.ndarray:
        f = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=self.chirp_time))
        c = self.nfft // 2
        return f[c - self.n_doppler // 2: c + self.n_doppler // 2]


def simulate_beat_signal(cfg: RadarConfig, positions: np.ndarray, weights: np.ndarray,
                         tx: np.ndarray, rx: np.ndarray, snr_db: float,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the complex beat signal, shape (n_chirps, n_fast).

    positions : (n_chirps, K, 3) scatterer positions per chirp [m] (body and clutter)
    weights   : (K,) amplitude weights
    tx, rx    : transmitter / receiver positions [m]; tx == rx for monostatic
    snr_db    : per-sample SNR of a unit-weight scatterer at 5 m (monostatic)
    """
    rng = rng or np.random.default_rng()
    pos = positions
    w = weights

    r_tx = np.linalg.norm(pos - tx, axis=-1)          # (M, K)
    r_rx = np.linalg.norm(pos - rx, axis=-1)
    path = r_tx + r_rx
    amp = w[None, :] * 25.0 / np.maximum(r_tx * r_rx, 0.25)

    n = np.arange(cfg.n_fast) / cfg.n_fast            # fast-time in units of chirp time
    beat_cycles = cfg.slope * path / C0 * cfg.chirp_time   # beat frequency x chirp time
    carrier_phase = 2 * np.pi * path / cfg.wavelength
    phase = 2 * np.pi * beat_cycles[..., None] * n[None, None, :] + carrier_phase[..., None]
    sig = np.einsum("mk,mkn->mn", amp.astype(np.float32), np.exp(1j * phase.astype(np.float32)))

    sigma = 10 ** (-snr_db / 20.0)
    noise = (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) * sigma / np.sqrt(2)
    return (sig + noise).astype(np.complex64)


def range_profiles(cfg: RadarConfig, beat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Range FFT with Hann window. Returns (profiles (M, n_fast), range axis [m]).

    The range axis is the *equivalent one-way* range, i.e. path length / 2.
    """
    win = np.hanning(cfg.n_fast).astype(np.float32)
    prof = np.fft.fft(beat * win[None, :], axis=1)
    beat_freq = np.arange(cfg.n_fast) / cfg.chirp_time
    rng_axis = beat_freq * C0 / cfg.slope / 2.0
    return prof, rng_axis


def micro_doppler(cfg: RadarConfig, beat: np.ndarray) -> np.ndarray:
    """Beat signal -> normalised micro-Doppler spectrogram (n_doppler, n_frames).

    Steps: range FFT -> MTI (remove static clutter) -> range gating ->
    STFT per range bin -> incoherent (power) sum over gated bins -> dB,
    normalised to [0, 1]. Positive Doppler means the path length is
    shrinking (approaching).

    Note: summing the *complex* range bins over a wide gate would cancel the
    target (the sum of all bins of a windowed FFT equals the first windowed
    sample, which is zero for a Hann window), so powers are summed instead.
    """
    prof, rng_axis = range_profiles(cfg, beat)
    prof = prof - prof.mean(axis=0, keepdims=True)            # MTI: remove DC per range bin
    gate = (rng_axis >= cfg.range_gate[0]) & (rng_axis <= cfg.range_gate[1])
    slow = np.conj(prof[:, gate])                              # approaching -> positive Doppler

    _, _, Z = stft(slow, fs=cfg.prf, window="hann", nperseg=cfg.win,
                   noverlap=cfg.win - cfg.hop, nfft=cfg.nfft, boundary=None,
                   padded=False, return_onesided=False, axis=0)
    # Z: (nfft, n_gated_bins, n_frames)
    power = (np.abs(Z) ** 2).sum(axis=1)
    power = np.fft.fftshift(power, axes=0)
    c = cfg.nfft // 2
    power = power[c - cfg.n_doppler // 2: c + cfg.n_doppler // 2]
    p_db = 10 * np.log10(power + 1e-12)
    p_db -= p_db.max()
    p_db = np.clip(p_db, -cfg.dyn_range_db, 0.0)
    return ((p_db + cfg.dyn_range_db) / cfg.dyn_range_db).astype(np.float32)
