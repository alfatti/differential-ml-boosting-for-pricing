"""Huge-Savine differential loss with per-coordinate sensitivity selection.

Key design choice: the user controls *which* sensitivity coordinates enter the
loss, and *how* they are weighted. This is the knob you asked for.

The base loss is:

    L = ||V - V_hat||^2 / s_V^2
      + lambda * sum_{k in active}  w_k * ||dV_k - dV_hat_k||^2 / s_k^2

where:
    - active = set of selected coordinate indices (e.g. {0, 4} = {S0, sigma})
    - s_V, s_k are robust scale normalizers (typically std of the training labels)
    - w_k are per-coordinate weights (default 1.0)
    - lambda is the global price-vs-differential balance

Scales are computed once from training labels and held fixed. We use std rather
than range to be robust to outliers in the AAD differentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


@dataclass
class SensitivitySpec:
    """User-facing spec for which sensitivities to include in the loss.

    Examples:
        # All sensitivities, equal weight (Huge-Savine default behavior)
        SensitivitySpec(active_indices=None)

        # Only spot and vol (delta + vega), with vega upweighted
        SensitivitySpec(active_indices=[0, 4], weights={4: 5.0})

        # Skip differential supervision entirely
        SensitivitySpec(active_indices=[])

    Attributes:
        active_indices: indices of X-coordinates whose AAD differentials enter
            the loss. None => all coordinates. Empty list => price-only training.
        weights: per-index weight overrides; defaults to 1.0 for any active
            index not listed.
    """
    active_indices: Optional[Sequence[int]] = None
    weights: dict[int, float] = field(default_factory=dict)

    def resolve(self, d_X: int) -> tuple[list[int], torch.Tensor]:
        """Return (sorted active index list, weight tensor of same length)."""
        if self.active_indices is None:
            active = list(range(d_X))
        else:
            active = sorted(set(int(i) for i in self.active_indices))
            for i in active:
                if not (0 <= i < d_X):
                    raise ValueError(f"active index {i} out of range [0, {d_X})")
        w = torch.tensor([float(self.weights.get(i, 1.0)) for i in active])
        return active, w


class DifferentialLoss(torch.nn.Module):
    """Huge-Savine MSE on prices + selected differentials, with fixed scales.

    Args:
        d_X: input dimension.
        sensitivity_spec: which sensitivities to use.
        lam: lambda, global balance between price loss and differential loss.
        scale_V: optional override for price scale. If None, computed from
            labels via `fit_scales`.
        scale_dV: optional override for per-coordinate dV scales (length d_X).

    Workflow:
        loss_fn = DifferentialLoss(d_X=5, sensitivity_spec=spec, lam=1.0)
        loss_fn.fit_scales(y_price_train, y_diff_train)   # call once
        ... then use loss_fn(V_hat, dV_hat, y_price, y_diff) in the training loop.
    """

    def __init__(
        self,
        d_X: int,
        sensitivity_spec: SensitivitySpec | None = None,
        lam: float = 1.0,
    ):
        super().__init__()
        self.d_X = d_X
        self.spec = sensitivity_spec or SensitivitySpec()
        self.lam = float(lam)

        active, w = self.spec.resolve(d_X)
        self.active = active  # list[int]
        # register weights as buffer so they move with .to(device)
        self.register_buffer("weights", w)

        # Scales -- set by fit_scales(); placeholders for now.
        self.register_buffer("scale_V", torch.tensor(1.0))
        self.register_buffer("scale_dV", torch.ones(d_X))
        self._scales_fitted = False

    # ----- scale fitting -------------------------------------------------------

    def fit_scales(
        self,
        y_price: torch.Tensor,
        y_diff: torch.Tensor,
        eps: float = 1e-8,
    ) -> None:
        """Compute per-coordinate std normalizers from training labels.

        Following Huge-Savine, we normalize by std of the *labels* so that the
        loss is dimensionally consistent across heterogeneous coordinates
        (e.g. delta ~ O(1), vega ~ O(spot * sqrt(T))).
        """
        sV = y_price.detach().std().clamp_min(eps)
        sD = y_diff.detach().std(dim=0).clamp_min(eps)  # (d_X,)
        self.scale_V = sV.to(self.scale_V.device, self.scale_V.dtype)
        self.scale_dV = sD.to(self.scale_dV.device, self.scale_dV.dtype)
        self._scales_fitted = True

    # ----- forward -------------------------------------------------------------

    def forward(
        self,
        V_hat: torch.Tensor,
        dV_hat: torch.Tensor,
        y_price: torch.Tensor,
        y_diff: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the loss and return a dict of components for logging.

        Args:
            V_hat:  (N,) model price prediction
            dV_hat: (N, d_X) model differential prediction
            y_price:(N,) target prices (path-averaged MC)
            y_diff: (N, d_X) target differentials (AAD)

        Returns:
            Dict with keys 'loss', 'price_mse', 'diff_mse', 'diff_mse_per_coord'.
        """
        if not self._scales_fitted:
            raise RuntimeError("Call fit_scales(...) before computing loss.")

        # Price term.
        price_resid = (V_hat - y_price) / self.scale_V
        price_mse = price_resid.pow(2).mean()

        # Differential term (only over active indices).
        if len(self.active) > 0:
            idx = torch.as_tensor(self.active, device=V_hat.device, dtype=torch.long)
            dV_pred_sel = dV_hat.index_select(1, idx)        # (N, k)
            dV_targ_sel = y_diff.index_select(1, idx)        # (N, k)
            scale_sel = self.scale_dV.index_select(0, idx)   # (k,)
            diff_resid = (dV_pred_sel - dV_targ_sel) / scale_sel  # broadcast over N
            # weighted MSE per coordinate, then sum
            per_coord = (diff_resid.pow(2).mean(dim=0)) * self.weights  # (k,)
            diff_mse = per_coord.sum()
        else:
            per_coord = torch.zeros(0, device=V_hat.device, dtype=V_hat.dtype)
            diff_mse = torch.tensor(0.0, device=V_hat.device, dtype=V_hat.dtype)

        loss = price_mse + self.lam * diff_mse
        return {
            "loss": loss,
            "price_mse": price_mse.detach(),
            "diff_mse": diff_mse.detach(),
            "diff_mse_per_coord": per_coord.detach(),
        }
