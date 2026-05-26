"""Abstract base for MC simulators.

A simulator generates pathwise prices and pathwise differentials via torch AAD.
Model-specific dynamics live in concrete subclasses; everything downstream is
model-agnostic.

Contract:
    simulate(X, payoff_fn) -> (V_path, dV_path)
        X         : (N, d_X) tensor of contractual/market params
        payoff_fn : callable mapping (S_T, X) -> per-path payoff
        V_path    : (N,)   pathwise (already discounted) payoff value
        dV_path   : (N, d_X) pathwise AAD differential of V w.r.t. X

The same Brownian paths can be reused across calls by passing a fixed
generator/seed; this is essential for the curriculum (we want stage-to-stage
differences to come only from payoff sharpening, not MC noise).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch


PayoffFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Simulator(ABC):
    """Abstract Monte Carlo simulator with pathwise AAD differentials."""

    # Names of the coordinates of X, in order. Subclasses must set this.
    param_names: tuple[str, ...] = ()

    # Index of the "vol-like" coordinate used for vega-adjusted bps eval.
    # Subclasses must set this.
    vega_index: int = -1

    @abstractmethod
    def simulate(
        self,
        X: torch.Tensor,
        payoff_fn: PayoffFn,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate pathwise prices and AAD differentials.

        Args:
            X: (N, d_X) contractual/market parameters, one row per training sample.
            payoff_fn: maps (S_T, X) -> (N,) per-sample (already path-averaged or
                pathwise) payoff. The implementation decides whether to average
                paths internally or expose pathwise outputs.
            n_paths: number of MC paths per sample.
            seed: optional seed for reproducibility / path reuse across stages.

        Returns:
            V:  (N,)   pathwise (or path-averaged) price estimate per sample
            dV: (N, d_X) AAD differential of V w.r.t. X, per sample
        """
        ...

    @property
    def dim_X(self) -> int:
        return len(self.param_names)
