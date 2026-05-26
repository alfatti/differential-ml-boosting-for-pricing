"""Black-Scholes MC simulator with pathwise AAD differentials.

Parameter layout: X = (S0, K, T, r, sigma). The terminal stock is

    S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

with Z ~ N(0,1). For training-data generation we draw n_paths samples per row
and average the (discounted) payoff. AAD then gives us d(V)/d(X) per row.

Note: this is a one-step exact GBM simulator (single time step from 0 to T).
For path-dependent payoffs you'd replace this with a stepped simulator; the
external interface stays the same.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import PayoffFn, Simulator


class BlackScholesSimulator(Simulator):
    """Single-step GBM simulator. X = (S0, K, T, r, sigma)."""

    param_names = ("S0", "K", "T", "r", "sigma")
    # Sigma is the vol coordinate -> vega index.
    vega_index = 4

    # Column indices for convenience.
    S0_IDX, K_IDX, T_IDX, R_IDX, SIGMA_IDX = 0, 1, 2, 3, 4

    def __init__(self, antithetic: bool = True):
        """Args:
            antithetic: use antithetic variates to halve MC variance.
        """
        self.antithetic = antithetic

    def simulate(
        self,
        X: torch.Tensor,
        payoff_fn: PayoffFn,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate (V, dV/dX) per row of X.

        Args:
            X: (N, 5) with columns (S0, K, T, r, sigma). dtype float32 or 64.
            payoff_fn: maps (S_T, X) -> (N, n_paths) per-path payoff (undiscounted).
            n_paths: paths per row.
            seed: optional seed for the Brownian draws.

        Returns:
            V:  (N,)   path-averaged discounted price per row
            dV: (N, 5) AAD differential of V w.r.t. each column of X
        """
        if X.shape[1] != 5:
            raise ValueError(f"BS expects 5-dim X (S0,K,T,r,sigma), got {X.shape[1]}")

        N = X.shape[0]
        device, dtype = X.device, X.dtype

        # We need d/dX, so X must require grad. Caller may pass either.
        X_req = X.detach().clone().requires_grad_(True)

        # Brownian draws. We do NOT make Z require grad; pathwise sensitivities
        # come from the chain rule through S_T's dependence on (S0,T,r,sigma).
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        if self.antithetic:
            half = n_paths // 2
            Z_half = torch.randn(N, half, device=device, dtype=dtype, generator=gen)
            Z = torch.cat([Z_half, -Z_half], dim=1)
            # If n_paths is odd, append one more independent draw.
            if 2 * half < n_paths:
                Z_extra = torch.randn(N, 1, device=device, dtype=dtype, generator=gen)
                Z = torch.cat([Z, Z_extra], dim=1)
        else:
            Z = torch.randn(N, n_paths, device=device, dtype=dtype, generator=gen)

        S0    = X_req[:, self.S0_IDX].unsqueeze(-1)
        T     = X_req[:, self.T_IDX].unsqueeze(-1)
        r     = X_req[:, self.R_IDX].unsqueeze(-1)
        sigma = X_req[:, self.SIGMA_IDX].unsqueeze(-1)

        sqrtT = torch.sqrt(T)
        drift = (r - 0.5 * sigma * sigma) * T
        diff  = sigma * sqrtT * Z
        S_T = S0 * torch.exp(drift + diff)  # (N, n_paths)

        # Undiscounted payoff (smoothed or true) then discount with exp(-rT).
        payoff = payoff_fn(S_T, X_req)            # (N, n_paths)
        discount = torch.exp(-r * T)              # (N, 1)
        V_paths = discount * payoff               # (N, n_paths)
        V = V_paths.mean(dim=1)                   # (N,)

        # AAD: differentiate the *summed* price w.r.t. X_req. Each row's gradient
        # is independent, so summing is equivalent to per-row jacobian rows.
        (dV,) = torch.autograd.grad(
            outputs=V.sum(),
            inputs=X_req,
            create_graph=False,
            retain_graph=False,
        )
        return V.detach(), dV.detach()
