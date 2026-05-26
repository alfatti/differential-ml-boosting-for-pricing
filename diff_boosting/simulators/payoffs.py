"""Payoffs (model-agnostic).

The smoothed call uses softplus: g_beta(S, K) = (1/beta) * log(1 + exp(beta*(S-K))).
As beta -> inf, g_beta -> (S-K)+. Smaller beta = smoother, larger bias upward.

The smoothing width (in units of S-K) is ~ 1/beta; we usually parameterize by
beta_width = 1/beta in % of spot to make the schedule interpretable.
"""

from __future__ import annotations

import torch


def call_payoff(S_T: torch.Tensor, X: torch.Tensor, strike_index: int = 1) -> torch.Tensor:
    """True European call payoff: (S_T - K)+.

    Args:
        S_T: (N,) or (N, n_paths) terminal stock values
        X: (N, d_X) parameter rows; strike is at column `strike_index`
        strike_index: column of X holding the strike

    Returns:
        Tensor of same shape as S_T with payoff per path.
    """
    K = X[:, strike_index]
    if S_T.dim() == 2:
        K = K.unsqueeze(-1)  # broadcast over paths
    return torch.clamp(S_T - K, min=0.0)


def softplus_call_payoff(
    S_T: torch.Tensor,
    X: torch.Tensor,
    beta: float,
    strike_index: int = 1,
) -> torch.Tensor:
    """Softplus-smoothed call payoff: (1/beta) * log(1 + exp(beta*(S_T - K))).

    Numerically stabilized via torch.nn.functional.softplus, which uses the
    standard threshold trick to avoid overflow for large beta*(S-K).

    Args:
        S_T: (N,) or (N, n_paths) terminal stock values
        X: (N, d_X) parameter rows; strike is at column `strike_index`
        beta: positive smoothing parameter. Larger beta = sharper. beta=inf
            recovers the true payoff.
        strike_index: column of X holding the strike

    Returns:
        Tensor of same shape as S_T with smoothed payoff per path.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    K = X[:, strike_index]
    if S_T.dim() == 2:
        K = K.unsqueeze(-1)
    # softplus(x) = log(1 + exp(x)); we want softplus(beta*(S-K)) / beta
    return torch.nn.functional.softplus(beta * (S_T - K)) / beta


def make_payoff_fn(beta: float | None, strike_index: int = 1):
    """Factory: build a payoff_fn closure for a given beta.

    Pass beta=None for the true (non-smoothed) payoff. This is what stage n
    of the curriculum uses if you want the final stage to fit the true kink.
    For our 0.1 bps target we recommend a large-but-finite beta_max instead.
    """
    if beta is None:
        def _fn(S_T, X):
            return call_payoff(S_T, X, strike_index=strike_index)
        return _fn
    else:
        def _fn(S_T, X):
            return softplus_call_payoff(S_T, X, beta=beta, strike_index=strike_index)
        return _fn
