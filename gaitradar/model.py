"""Heteroscedastic CNN for gait parameter regression and a deep-ensemble wrapper.

Each network predicts, for every target (speed, cadence), a mean and a
log-variance, i.e. a Gaussian predictive distribution that captures
*aleatoric* uncertainty (noise, ambiguity in the measurement). Training
several independently initialised networks (a deep ensemble) adds
*epistemic* uncertainty (what the model does not know, e.g. under
conditions it was not trained on):

    mu  = mean_m mu_m
    var = mean_m sigma_m^2 + var_m mu_m      (law of total variance)

Loss: beta-NLL (Seitzer et al., ICLR 2022) with beta = 0.5, which keeps the
mean fit from being neglected on high-variance samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GaitNet(nn.Module):
    """Input: (B, C, n_doppler, n_frames) spectrogram(s). C = number of radar channels."""

    def __init__(self, in_ch: int = 1, width: int = 16, n_targets: int = 2, dropout: float = 0.1):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(in_ch, w), conv_block(w, 2 * w), conv_block(2 * w, 4 * w), conv_block(4 * w, 4 * w),
        )
        self.head = nn.Sequential(
            nn.Linear(8 * w, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 2 * n_targets),
        )
        self.n_targets = n_targets

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        out = self.head(f)
        mu, log_var = out[:, : self.n_targets], out[:, self.n_targets:]
        return mu, log_var.clamp(-8.0, 4.0)


def beta_nll(mu, log_var, y, beta: float = 0.5):
    var = log_var.exp()
    nll = 0.5 * (log_var + (y - mu) ** 2 / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()


@torch.no_grad()
def ensemble_predict(models, x: torch.Tensor, batch: int = 64) -> dict:
    """Ensemble prediction in standardised units.

    Returns a dict with ``mu``, ``var`` (total), ``aleatoric`` (mean of the
    members' variances) and ``epistemic`` (variance of the members' means),
    each of shape (N, n_targets).
    """
    mus, vars_ = [], []
    for m in models:
        m.eval()
        mu_l, var_l = [], []
        for i in range(0, len(x), batch):
            mu, lv = m(x[i:i + batch])
            mu_l.append(mu)
            var_l.append(lv.exp())
        mus.append(torch.cat(mu_l))
        vars_.append(torch.cat(var_l))
    mus = torch.stack(mus).numpy()        # (M, N, T)
    vars_ = torch.stack(vars_).numpy()
    aleatoric, epistemic = vars_.mean(0), mus.var(0)
    return dict(mu=mus.mean(0), var=aleatoric + epistemic, aleatoric=aleatoric, epistemic=epistemic)
