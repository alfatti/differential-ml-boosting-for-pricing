"""Stacked-residual DML architecture.

Stage 0:    V^(0)(X) = MLP_0(X)
Stage i>0:  V^(i)(X) = V^(i-1)(X) + |alpha_i| * MLP_i([X, V^(i-1)(X)])

Each MLP outputs a scalar. The cumulative price V^(i) is differentiable end-to-end
in X, so dV^(i)/dX is obtained via autograd at training/eval time. We deliberately
do NOT use functorch.jacrev here -- a single autograd.grad call on the summed
output is faster and gives us exactly what Huge-Savine needs.

Conventions:
    - Inputs X are assumed already normalized (mean 0, std 1 per column) by the
      caller. Denormalization for prices/diffs happens in the trainer.
    - Each stage owns its own parameters; greedy training freezes earlier stages.
    - alpha is stored as an unconstrained parameter and used as |alpha| so the
      sign is irrelevant and the gradient flows cleanly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def _mlp(d_in: int, d_out: int, hidden: Sequence[int], activation: str = "softplus") -> nn.Sequential:
    """Build an MLP. We use softplus by default so the model is C^infty in X,
    which makes second derivatives well-behaved (matters if you ever fold gamma
    into the loss).
    """
    act_map = {
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation '{activation}'. Choose from {list(act_map)}.")
    Act = act_map[activation]

    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), Act()]
        prev = h
    layers += [nn.Linear(prev, d_out)]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Thin wrapper around `_mlp` so submodules show up cleanly in state_dict."""

    def __init__(self, d_in: int, d_out: int, hidden: Sequence[int], activation: str = "softplus"):
        super().__init__()
        self.net = _mlp(d_in, d_out, hidden, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StackedResidualDML(nn.Module):
    """Stacked residual DML pricer.

    Args:
        d_X: input dimension (e.g. 5 for BS).
        n_stages: total number of stages = 1 (baseline) + (n_stages - 1) residual blocks.
            Pass n_stages=1 to recover a vanilla DML net (no boosting).
        hidden_baseline: hidden layer sizes for stage 0.
        hidden_residual: hidden layer sizes for residual blocks.
        activation: nonlinearity. Softplus default keeps the model C^infty.
        alpha_init: initial value of the (unconstrained) alpha scalar per
            residual block. Small => correction starts small.

    Forward returns the cumulative prediction V^(stage). By default `stage=-1`
    means "all stages combined" (i.e. n_stages - 1). To get intermediate
    predictions for diagnostics, pass `stage=k`.
    """

    def __init__(
        self,
        d_X: int,
        n_stages: int = 3,
        hidden_baseline: Sequence[int] = (64, 64, 64),
        hidden_residual: Sequence[int] = (32, 32),
        activation: str = "softplus",
        alpha_init: float = 0.1,
    ):
        super().__init__()
        if n_stages < 1:
            raise ValueError("n_stages must be >= 1")
        self.d_X = d_X
        self.n_stages = n_stages

        # Stage 0: baseline.
        self.baseline = MLP(d_X, 1, hidden_baseline, activation=activation)

        # Stages 1..n_stages-1: residual blocks taking [X, V_prev].
        self.residual_blocks = nn.ModuleList([
            MLP(d_X + 1, 1, hidden_residual, activation=activation)
            for _ in range(n_stages - 1)
        ])

        # One unconstrained alpha per residual block.
        self.alphas_raw = nn.Parameter(
            torch.full((max(n_stages - 1, 1),), float(alpha_init))
        )

    # ---- introspection helpers ------------------------------------------------

    def alpha(self, i: int) -> torch.Tensor:
        """|alpha_i| for residual block i (1-indexed in math, 0-indexed here)."""
        return self.alphas_raw[i].abs()

    def stage_parameters(self, stage: int):
        """Iterator over parameters owned by stage `stage` only (for greedy training).

        stage=0 -> baseline params.
        stage>=1 -> residual block (stage-1) params + alpha (stage-1).
        """
        if stage == 0:
            yield from self.baseline.parameters()
        else:
            idx = stage - 1
            yield from self.residual_blocks[idx].parameters()
            # alpha is a slice of a Parameter; we yield the whole tensor and
            # mask gradients via freezing in the trainer.
            yield self.alphas_raw

    def freeze_through(self, stage: int) -> None:
        """Freeze all params for stages 0..stage (inclusive). Used between
        greedy stage trainings."""
        if stage >= 0:
            for p in self.baseline.parameters():
                p.requires_grad_(False)
        for i in range(min(stage, len(self.residual_blocks) - 1) + 1 - 1):
            # freeze residual blocks strictly below `stage`
            for p in self.residual_blocks[i].parameters():
                p.requires_grad_(False)
        # Note: alphas are a single Parameter tensor; we do NOT zero out
        # gradients of frozen entries here. The trainer handles that by
        # masking the gradient after backward().

    # ---- forward --------------------------------------------------------------

    def forward(self, X: torch.Tensor, stage: Optional[int] = None) -> torch.Tensor:
        """Compute cumulative price V^(stage) at the requested stage.

        Args:
            X: (N, d_X) input.
            stage: which stage's cumulative output to return. None => last stage.
                Must be in [0, n_stages - 1].

        Returns:
            (N,) tensor of prices.
        """
        if stage is None:
            stage = self.n_stages - 1
        if not (0 <= stage <= self.n_stages - 1):
            raise ValueError(f"stage must be in [0, {self.n_stages - 1}], got {stage}")

        V = self.baseline(X).squeeze(-1)  # (N,)
        for i in range(stage):
            inp = torch.cat([X, V.unsqueeze(-1)], dim=-1)
            correction = self.residual_blocks[i](inp).squeeze(-1)
            V = V + self.alpha(i) * correction
        return V

    def predict_with_diff(
        self,
        X: torch.Tensor,
        stage: Optional[int] = None,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (V^(stage), dV^(stage)/dX) using autograd.

        Args:
            X: (N, d_X). Will be made to require_grad internally if it doesn't.
            stage: stage index (default last).
            create_graph: True if you need higher-order derivatives (e.g. gamma
                in the loss); False otherwise (saves memory + time).

        Returns:
            V:  (N,)   prices.
            dV: (N, d_X) input-gradients.
        """
        X_req = X if X.requires_grad else X.detach().clone().requires_grad_(True)
        V = self.forward(X_req, stage=stage)
        (dV,) = torch.autograd.grad(
            outputs=V.sum(),
            inputs=X_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        return V, dV
