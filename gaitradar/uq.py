"""Metrics for point accuracy and uncertainty quality."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr


def mae(y, mu):
    return float(np.mean(np.abs(y - mu)))


def rmse(y, mu):
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def gaussian_nll(y, mu, sigma):
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2))


def coverage(y, mu, sigma, level: float) -> float:
    """Fraction of targets inside the central ``level`` prediction interval."""
    z = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(y - mu) <= z * sigma))


def calibration_curve(y, mu, sigma, levels=None):
    levels = np.linspace(0.05, 0.95, 19) if levels is None else levels
    return levels, np.array([coverage(y, mu, sigma, p) for p in levels])


def calibration_error(y, mu, sigma) -> float:
    """Mean absolute gap between nominal and observed interval coverage."""
    levels, cov = calibration_curve(y, mu, sigma)
    return float(np.mean(np.abs(cov - levels)))


def interval_width(sigma, level: float) -> float:
    return float(np.mean(2 * norm.ppf(0.5 + level / 2) * sigma))


def error_uncertainty_corr(y, mu, sigma) -> float:
    """Spearman rank correlation between |error| and predicted sigma."""
    return float(spearmanr(np.abs(y - mu), sigma).correlation)


def selective_mae(y, mu, sigma, keep=None):
    """MAE when only the most confident fraction ``keep`` of predictions is kept."""
    keep = np.linspace(0.3, 1.0, 15) if keep is None else keep
    order = np.argsort(sigma)
    err = np.abs(y - mu)[order]
    return keep, np.array([err[: max(1, int(round(k * len(err))))].mean() for k in keep])


def fit_sigma_scale(y, mu, sigma) -> float:
    """Closed-form variance scaling that minimises Gaussian NLL (post-hoc recalibration)."""
    return float(np.sqrt(np.mean(((y - mu) / sigma) ** 2)))


def summary(y, mu, sigma, level: float = 0.9) -> dict:
    return dict(
        mae=mae(y, mu), rmse=rmse(y, mu), nll=gaussian_nll(y, mu, sigma),
        coverage90=coverage(y, mu, sigma, level), width90=interval_width(sigma, level),
        cal_error=calibration_error(y, mu, sigma), spearman=error_uncertainty_corr(y, mu, sigma),
        mean_sigma=float(np.mean(sigma)),
    )
