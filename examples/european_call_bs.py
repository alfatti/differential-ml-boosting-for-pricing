"""End-to-end demo: differential boosting on European calls under Black-Scholes.

Pipeline:
    1. Sample contractual parameters X = (S0, K, T, r, sigma) on a wide grid.
    2. For each curriculum stage, simulate (V, dV/dX) at the stage's beta via
       AAD on a BS one-step GBM simulator.
    3. Greedy-train the stacked-residual DML.
    4. Evaluate vega-adjusted bps error against BS closed-form, bucketed by
       moneyness, paying particular attention to the ITM region.
"""

from __future__ import annotations

import torch

from diff_boosting.simulators import BlackScholesSimulator
from diff_boosting.models import StackedResidualDML
from diff_boosting.training import (
    DifferentialBoostingTrainer,
    CurriculumSchedule,
    SensitivitySpec,
)
from diff_boosting.training.trainer import StageConfig
from diff_boosting.evaluation import (
    bs_call_price,
    bs_call_greeks,
    vega_adjusted_bps_error,
    error_by_moneyness_bucket,
)


def sample_X(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Uniform sampling over a realistic parameter cube.

    Layout: (S0, K, T, r, sigma). We vary all of S0, K, T, sigma; r is held
    fixed (typical for a pricer trained at a given rate environment), and the
    trainer will warn if you select dV/dr as an active sensitivity.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 80.0 + 40.0 * torch.rand(n, device=device, generator=g)   # 80 .. 120
    K = 80.0 + 40.0 * torch.rand(n, device=device, generator=g)    # 80 .. 120
    T = 0.05 + 1.95 * torch.rand(n, device=device, generator=g)    # 0.05 .. 2y
    r = 0.02 * torch.ones(n, device=device)
    sigma = 0.10 + 0.30 * torch.rand(n, device=device, generator=g)  # 10 .. 40%
    return torch.stack([S0, K, T, r, sigma], dim=1)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- 1. Data ------------------------------------------------------------
    n_train, n_val = 8_000, 2_000
    X_train = sample_X(n_train, device, seed=1)
    X_val = sample_X(n_val, device, seed=2)

    # ---- 2. Simulator, model, trainer --------------------------------------
    sim = BlackScholesSimulator(antithetic=True)

    n_stages = 3
    model = StackedResidualDML(
        d_X=sim.dim_X,
        n_stages=n_stages,
        hidden_baseline=(64, 64, 64),
        hidden_residual=(32, 32),
        activation="softplus",
        alpha_init=0.1,
    )

    # Sensitivity control: include delta (S0) and vega (sigma) with vega upweighted,
    # ignore K, T, r sensitivities (we have less use for them at training time).
    spec = SensitivitySpec(
        active_indices=[sim.S0_IDX, sim.SIGMA_IDX],
        weights={sim.SIGMA_IDX: 5.0},
    )

    trainer = DifferentialBoostingTrainer(
        model=model,
        simulator=sim,
        sensitivity_spec=spec,
        lam=1.0,
        device=device,
        normalize_inputs=True,
    )

    schedule = CurriculumSchedule(
        n_stages=n_stages,
        width_init=5.0,   # smoothing over $5 at stage 0
        width_final=0.05, # smoothing over $0.05 at final stage
        p=2.0,
    )

    stage_cfgs = [
        StageConfig(n_epochs=2000, lr=1e-3, patience=200, log_every=200),
        StageConfig(n_epochs=2000, lr=5e-4, patience=200, log_every=200),
        StageConfig(n_epochs=2000, lr=2e-4, patience=200, log_every=200),
    ]

    # ---- 3. Train -----------------------------------------------------------
    trainer.train_with_simulator(
        X_train=X_train,
        X_val=X_val,
        schedule=schedule,
        stage_configs=stage_cfgs,
        n_paths=8192,
        seed=42,
    )

    # ---- 4. Evaluate against BS closed-form --------------------------------
    n_test = 5_000
    X_test = sample_X(n_test, device, seed=3)
    S0, K, T, r, sigma = X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3], X_test[:, 4]
    V_ref = bs_call_price(S0, K, T, r, sigma)
    greeks = bs_call_greeks(S0, K, T, r, sigma)
    vega = greeks["vega"]

    # Predictions at every stage for diagnostics.
    print("\n=== Test-set vega-adjusted bps error ===")
    log_moneyness = torch.log(S0 / K).cpu()
    for stage in range(n_stages):
        V_hat = trainer.predict(X_test, stage=stage).to(device)
        err = vega_adjusted_bps_error(V_hat, V_ref, vega).cpu()
        print(f"stage {stage}: mean {err.mean():.3f} bps | "
              f"median {err.median():.3f} bps | "
              f"p95 {err.quantile(0.95):.3f} bps | "
              f"max {err.max():.3f} bps")

    # Final stage by moneyness bucket.
    V_hat = trainer.predict(X_test, stage=n_stages - 1).to(device)
    err = vega_adjusted_bps_error(V_hat, V_ref, vega).cpu()
    print("\n=== Final stage: error by log-moneyness bucket ===")
    buckets = error_by_moneyness_bucket(log_moneyness, err)
    print(f"{'bucket':>18}  {'count':>6}  {'mean':>10}  {'median':>10}  {'p95':>10}")
    for label, (cnt, mean, med, p95) in buckets.items():
        print(f"{label:>18}  {cnt:>6}  {mean:>10.4f}  {med:>10.4f}  {p95:>10.4f}")

    print(f"\nLearned alphas: {[float(model.alpha(i).detach()) for i in range(n_stages - 1)]}")


if __name__ == "__main__":
    main()
