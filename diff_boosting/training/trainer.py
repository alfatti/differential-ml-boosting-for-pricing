"""Greedy stage-by-stage trainer for the stacked-residual DML.

Curriculum: the payoff smoothing parameter beta increases through stages,
0 < beta_0 < beta_1 < ... < beta_n. Per the prior discussion we keep beta_n
*finite* (a controlled smoothing bias under 0.1 bps) rather than the true kink.

For each stage:
  1. Generate (or supply) labels at beta_i.
  2. Freeze all parameters from earlier stages.
  3. Train this stage's parameters (sub-network + alpha) on the Huge-Savine loss
     applied to the cumulative prediction V^(i).
  4. Early-stop on validation loss plateau.

Path reuse: the *same* Brownian seed is used across stages so the only signal
difference between labels at beta_i and beta_{i+1} is payoff sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from ..models.residual_dml import StackedResidualDML
from ..simulators.base import Simulator
from ..simulators.payoffs import make_payoff_fn
from .losses import DifferentialLoss, SensitivitySpec


@dataclass
class CurriculumSchedule:
    """Payoff-smoothing schedule across stages.

    The smoothing width in $-units is roughly w_i = 1/beta_i. We parameterize
    the schedule in width-space because it's interpretable: w_0 = "smooth over
    5% of spot", w_n = "smooth over 1 bp of spot", etc.

    The schedule is: w_i = w_init * (1 - (i/n)^p) + w_final * (i/n)^p with p>1
    so it bites mostly near the end (mirroring the paper's super-linear gamma).

    Attributes:
        n_stages: total stages (matches the model's n_stages).
        width_init: starting smoothing width in *absolute price units* (same
            units as S - K). For a call on spot ~ 100, width_init=5.0 means
            smoothing over $5 of moneyness.
        width_final: final smoothing width, > 0 to keep beta finite. Choose
            e.g. 0.01 (1 cent) for a tight pricer.
        p: superlinearity exponent (>1).
    """
    n_stages: int
    width_init: float = 5.0
    width_final: float = 0.01
    p: float = 2.0

    def beta(self, i: int) -> float:
        """Return beta_i for stage i in [0, n_stages-1]."""
        if not (0 <= i < self.n_stages):
            raise ValueError(f"stage {i} out of range")
        if self.n_stages == 1:
            # Single stage: just use width_final.
            return 1.0 / self.width_final
        frac = (i / (self.n_stages - 1)) ** self.p
        w = self.width_init * (1.0 - frac) + self.width_final * frac
        return 1.0 / w


@dataclass
class StageConfig:
    """Hyperparameters for training one stage."""
    n_epochs: int = 2000
    lr: float = 1e-3
    batch_size: Optional[int] = None  # None = full-batch
    patience: int = 200                # early-stop patience on val loss
    min_delta: float = 1e-6            # min improvement to reset patience
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    log_every: int = 100


@dataclass
class TrainingHistory:
    """Per-stage training logs."""
    stage_losses: list[list[float]] = field(default_factory=list)
    stage_val_losses: list[list[float]] = field(default_factory=list)
    stage_alphas: list[float] = field(default_factory=list)


class DifferentialBoostingTrainer:
    """Greedy trainer for StackedResidualDML.

    Args:
        model: the StackedResidualDML instance.
        simulator: a Simulator (BS, Heston, etc.) -- only used if you call
            `train_with_simulator`. If you pre-generate labels, you don't need it.
        sensitivity_spec: which sensitivities enter the loss.
        lam: differential vs price balance.
        device: torch device.
        normalize_inputs: if True, standardize X with train-set mean/std before
            feeding the model (recommended for stability).
    """

    def __init__(
        self,
        model: StackedResidualDML,
        simulator: Optional[Simulator] = None,
        sensitivity_spec: Optional[SensitivitySpec] = None,
        lam: float = 1.0,
        device: str | torch.device = "cpu",
        normalize_inputs: bool = True,
    ):
        self.model = model.to(device)
        self.simulator = simulator
        self.spec = sensitivity_spec or SensitivitySpec()
        self.lam = lam
        self.device = torch.device(device)
        self.normalize_inputs = normalize_inputs

        # input normalizer (set by fit)
        self._X_mean: Optional[torch.Tensor] = None
        self._X_std: Optional[torch.Tensor] = None

        self.history = TrainingHistory()

    # ---- input normalization --------------------------------------------------

    def fit_input_normalizer(self, X_train: torch.Tensor, const_tol: float = 1e-6) -> None:
        """Fit per-column mean/std for input normalization.

        Constant (or near-constant) columns are detected and given std=1 with
        mean=value so the normalized column is exactly zero. This avoids the
        garbage-in-garbage-out problem when 1/std blows up.
        """
        if self.normalize_inputs:
            mean = X_train.mean(dim=0)
            std = X_train.std(dim=0)
            constant_mask = std < const_tol
            # For constant columns, set std=1 -> normalized value = 0 (since X==mean).
            std = torch.where(constant_mask, torch.ones_like(std), std)
            self._X_mean = mean.to(self.device)
            self._X_std = std.to(self.device)
            self._constant_cols = constant_mask.to(self.device)
        else:
            self._X_mean = torch.zeros(X_train.shape[1], device=self.device)
            self._X_std = torch.ones(X_train.shape[1], device=self.device)
            self._constant_cols = torch.zeros(X_train.shape[1], dtype=torch.bool, device=self.device)

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self._X_mean) / self._X_std

    # ---- prediction in original units ----------------------------------------

    def predict(self, X: torch.Tensor, stage: Optional[int] = None) -> torch.Tensor:
        """Return cumulative price V^(stage)(X) on the original X scale."""
        self.model.eval()
        Xn = self._normalize(X.to(self.device))
        with torch.no_grad():
            return self.model(Xn, stage=stage).cpu()

    def predict_with_diff(
        self, X: torch.Tensor, stage: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (V, dV/dX) on the original X scale.

        Chain rule: if X_norm = (X - mu) / sigma, then dV/dX = (1/sigma) * dV/dX_norm.
        """
        self.model.eval()
        X_dev = X.to(self.device)
        Xn = self._normalize(X_dev).requires_grad_(True)
        V = self.model(Xn, stage=stage)
        (dV_norm,) = torch.autograd.grad(V.sum(), Xn, create_graph=False)
        # Undo input normalization on the gradient.
        dV = dV_norm / self._X_std  # (N, d_X) broadcast over rows
        return V.detach().cpu(), dV.detach().cpu()

    # ---- core: train one stage ------------------------------------------------

    def train_stage(
        self,
        stage: int,
        X_train: torch.Tensor,
        y_price_train: torch.Tensor,
        y_diff_train: torch.Tensor,
        X_val: torch.Tensor,
        y_price_val: torch.Tensor,
        y_diff_val: torch.Tensor,
        cfg: StageConfig,
    ) -> None:
        """Train parameters of `stage` only; earlier stages frozen.

        Labels here are stage-specific (generated at beta_stage). The loss
        compares against the cumulative prediction V^(stage)(X).
        """
        # --- freeze earlier stages -------------------------------------------
        # First, enable grad on everything in this stage's params; disable elsewhere.
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.stage_parameters(stage):
            p.requires_grad_(True)
        # Alphas are a single Parameter tensor; we need a mask so only this
        # stage's alpha gets a non-zero gradient.
        alpha_mask = torch.zeros_like(self.model.alphas_raw)
        if stage >= 1:
            alpha_mask[stage - 1] = 1.0

        # --- prepare data on device ------------------------------------------
        X_train_n = self._normalize(X_train.to(self.device))
        X_val_n = self._normalize(X_val.to(self.device))
        y_price_train = y_price_train.to(self.device)
        y_diff_train = y_diff_train.to(self.device)
        y_price_val = y_price_val.to(self.device)
        y_diff_val = y_diff_val.to(self.device)

        # Warn if user is asking the model to learn sensitivities w.r.t. an
        # input column that doesn't vary in training data -- the model cannot
        # see those directions and the gradient signal is uninformative.
        if stage == 0 and self.spec.active_indices is not None:
            for i in self.spec.active_indices:
                if bool(self._constant_cols[i]):
                    name = (
                        self.simulator.param_names[i]
                        if self.simulator is not None and i < len(self.simulator.param_names)
                        else f"col {i}"
                    )
                    print(
                        f"  WARNING: sensitivity coord {i} ('{name}') is selected but "
                        f"that column is constant in X_train. The model cannot learn "
                        f"this sensitivity. Consider removing it from active_indices."
                    )

        # --- loss (scales fitted on this stage's normalized-input labels) ----
        loss_fn = DifferentialLoss(
            d_X=self.model.d_X,
            sensitivity_spec=self.spec,
            lam=self.lam,
        ).to(self.device)

        # Differential labels expressed in normalized-input space:
        #   dV/dX_norm = sigma * dV/dX
        # (model outputs dV/dX_norm naturally because it eats normalized inputs)
        sigma_X = self._X_std.unsqueeze(0)  # (1, d_X)
        y_diff_train_n = y_diff_train * sigma_X
        y_diff_val_n = y_diff_val * sigma_X

        loss_fn.fit_scales(y_price_train, y_diff_train_n)

        # --- optimizer over trainable params only ----------------------------
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable params for stage {stage}.")
        opt = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # --- training loop ---------------------------------------------------
        best_val = float("inf")
        bad = 0
        epoch_losses: list[float] = []
        epoch_val_losses: list[float] = []

        for epoch in range(cfg.n_epochs):
            self.model.train()
            # Full-batch by default (datasets here are typically <= 100k rows).
            if cfg.batch_size is None or cfg.batch_size >= X_train_n.shape[0]:
                batches = [(X_train_n, y_price_train, y_diff_train_n)]
            else:
                perm = torch.randperm(X_train_n.shape[0], device=self.device)
                batches = []
                for s in range(0, X_train_n.shape[0], cfg.batch_size):
                    idx = perm[s : s + cfg.batch_size]
                    batches.append(
                        (X_train_n[idx], y_price_train[idx], y_diff_train_n[idx])
                    )

            ep_loss = 0.0
            for Xb, yp, yd in batches:
                Xb_req = Xb.detach().requires_grad_(True)
                V = self.model(Xb_req, stage=stage)
                (dV,) = torch.autograd.grad(V.sum(), Xb_req, create_graph=True)
                losses = loss_fn(V, dV, yp, yd)
                loss = losses["loss"]

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Mask alpha gradient so only this stage's entry updates.
                if self.model.alphas_raw.grad is not None:
                    self.model.alphas_raw.grad.mul_(alpha_mask)

                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
                ep_loss += float(loss.detach()) * Xb.shape[0]
            ep_loss /= X_train_n.shape[0]
            epoch_losses.append(ep_loss)

            # Validation.
            self.model.eval()
            X_val_req = X_val_n.detach().requires_grad_(True)
            V_val = self.model(X_val_req, stage=stage)
            (dV_val,) = torch.autograd.grad(V_val.sum(), X_val_req, create_graph=False)
            val_metrics = loss_fn(V_val, dV_val, y_price_val, y_diff_val_n)
            val_loss = float(val_metrics["loss"].detach())
            epoch_val_losses.append(val_loss)

            # Early stopping.
            if val_loss < best_val - cfg.min_delta:
                best_val = val_loss
                bad = 0
            else:
                bad += 1

            if epoch % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
                a = (
                    float(self.model.alpha(stage - 1).detach())
                    if stage >= 1
                    else float("nan")
                )
                print(
                    f"  [stage {stage}] epoch {epoch:5d} | "
                    f"train {ep_loss:.3e} | val {val_loss:.3e} | "
                    f"price_mse {float(val_metrics['price_mse']):.3e} | "
                    f"diff_mse {float(val_metrics['diff_mse']):.3e} | "
                    f"|alpha| {a:.4f}"
                )

            if bad >= cfg.patience:
                print(f"  [stage {stage}] early stop at epoch {epoch} (val plateau).")
                break

        self.history.stage_losses.append(epoch_losses)
        self.history.stage_val_losses.append(epoch_val_losses)
        if stage >= 1:
            self.history.stage_alphas.append(float(self.model.alpha(stage - 1).detach()))
        else:
            self.history.stage_alphas.append(float("nan"))

    # ---- driver: full curriculum ---------------------------------------------

    def train_with_simulator(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        schedule: CurriculumSchedule,
        stage_configs: Sequence[StageConfig],
        n_paths: int = 4096,
        seed: int = 0,
        strike_index: int = 1,
    ) -> None:
        """Run the full curriculum: for each stage i, generate labels at
        beta_i via the simulator, then train that stage.

        Args:
            X_train, X_val: input parameter rows.
            schedule: the curriculum schedule.
            stage_configs: one StageConfig per stage. If a single config is
                passed, it's reused.
            n_paths: MC paths per row, reused across stages (same seed).
            seed: base seed; we add stage offset 0 so paths are identical.
            strike_index: column of X holding the strike.
        """
        if self.simulator is None:
            raise RuntimeError("No simulator attached; pass `simulator=` to the trainer "
                               "or call `train_stage` directly with pre-generated labels.")
        if len(stage_configs) == 1:
            stage_configs = [stage_configs[0]] * self.model.n_stages
        if len(stage_configs) != self.model.n_stages:
            raise ValueError("len(stage_configs) must equal model.n_stages")

        self.fit_input_normalizer(X_train)

        for i in range(self.model.n_stages):
            beta_i = schedule.beta(i)
            print(f"\n=== Stage {i}/{self.model.n_stages - 1} | beta = {beta_i:.3f} "
                  f"(width = {1.0/beta_i:.4f}) ===")
            payoff = make_payoff_fn(beta=beta_i, strike_index=strike_index)
            y_p_tr, y_d_tr = self.simulator.simulate(X_train, payoff, n_paths=n_paths, seed=seed)
            y_p_va, y_d_va = self.simulator.simulate(X_val, payoff, n_paths=n_paths, seed=seed + 10_000)
            self.train_stage(
                stage=i,
                X_train=X_train,
                y_price_train=y_p_tr,
                y_diff_train=y_d_tr,
                X_val=X_val,
                y_price_val=y_p_va,
                y_diff_val=y_d_va,
                cfg=stage_configs[i],
            )
