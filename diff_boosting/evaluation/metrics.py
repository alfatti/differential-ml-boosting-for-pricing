"""Evaluation metrics.

Vega-adjusted bps error:
    err_bps = 10^4 * |V_hat - V| / Vega
Interpretation: how many bps of implied volatility you'd need to move to make
the price match. This is the trader-relevant metric for option pricers.

BS closed-forms here are *only* used as the evaluation oracle. Training never
sees them, keeping the pipeline model-agnostic.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ---- BS closed-form (oracle for evaluation only) -----------------------------


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _phi_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(
    S0: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Black-Scholes European call price."""
    sqrtT = torch.sqrt(T)
    d1 = (torch.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * _phi(d1) - K * torch.exp(-r * T) * _phi(d2)


def bs_call_greeks(
    S0: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return analytic delta and vega for a European call."""
    sqrtT = torch.sqrt(T)
    d1 = (torch.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    delta = _phi(d1)
    vega = S0 * _phi_pdf(d1) * sqrtT
    return {"delta": delta, "vega": vega}


# ---- model-agnostic metrics --------------------------------------------------


def vega_adjusted_bps_error(
    V_hat: torch.Tensor,
    V_ref: torch.Tensor,
    vega: torch.Tensor,
    vega_floor: float = 1e-4,
) -> torch.Tensor:
    """Vega-adjusted error in bps. Floors vega to avoid blow-up deep ITM/OTM
    where vega -> 0 (the metric becomes uninformative there)."""
    return 1e4 * (V_hat - V_ref).abs() / vega.clamp_min(vega_floor)


def error_by_moneyness_bucket(
    moneyness: torch.Tensor,
    err_bps: torch.Tensor,
    edges: Optional[list[float]] = None,
) -> dict[str, tuple[int, float, float, float]]:
    """Aggregate err_bps into moneyness buckets.

    Args:
        moneyness: per-sample moneyness (e.g. S0/K or log(S0/K)).
        err_bps: per-sample vega-adjusted bps error.
        edges: bucket edges. Default chosen for log-moneyness around 0.

    Returns:
        Dict bucket-label -> (count, mean, median, p95).
    """
    if edges is None:
        edges = [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4]
    out = {}
    edges = [-float("inf")] + list(edges) + [float("inf")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (moneyness >= lo) & (moneyness < hi)
        if mask.any():
            sub = err_bps[mask]
            label = f"[{lo:+.2f}, {hi:+.2f})"
            out[label] = (
                int(mask.sum()),
                float(sub.mean()),
                float(sub.median()),
                float(sub.quantile(0.95)),
            )
    return out
