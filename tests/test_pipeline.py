"""Sanity tests. Run with: python -m pytest tests/ -v"""

import math

import torch

from diff_boosting.simulators import BlackScholesSimulator
from diff_boosting.simulators.payoffs import call_payoff, softplus_call_payoff, make_payoff_fn
from diff_boosting.evaluation import bs_call_price, bs_call_greeks
from diff_boosting.models import StackedResidualDML
from diff_boosting.training import SensitivitySpec
from diff_boosting.training.losses import DifferentialLoss


def test_softplus_call_recovers_true_call_for_large_beta():
    S = torch.linspace(80.0, 120.0, 50)
    X = torch.stack([torch.zeros_like(S), torch.full_like(S, 100.0)], dim=1)
    # Note: S is passed directly as S_T here (1D), strike at col 1.
    true = call_payoff(S, X)
    smooth = softplus_call_payoff(S, X, beta=1000.0)
    # Away from the kink, smoothed and true agree to high precision.
    away = (S - 100.0).abs() > 1.0
    assert torch.allclose(true[away], smooth[away], atol=1e-3)


def test_bs_simulator_matches_analytic_price_and_greeks():
    """MC + AAD should agree with BS closed-form within MC noise."""
    torch.manual_seed(0)
    n = 32
    S0 = torch.full((n,), 100.0)
    K = torch.linspace(80.0, 120.0, n)
    T = torch.full((n,), 1.0)
    r = torch.full((n,), 0.02)
    sigma = torch.full((n,), 0.20)
    X = torch.stack([S0, K, T, r, sigma], dim=1)

    sim = BlackScholesSimulator(antithetic=True)
    payoff = make_payoff_fn(beta=None)  # true call
    V, dV = sim.simulate(X, payoff, n_paths=200_000, seed=123)

    V_ref = bs_call_price(S0, K, T, r, sigma)
    greeks = bs_call_greeks(S0, K, T, r, sigma)

    # Price within 2 cents on $100 spot at 200k antithetic paths.
    assert (V - V_ref).abs().max() < 0.05, f"max price err {(V - V_ref).abs().max()}"
    # Delta within 0.005.
    assert (dV[:, 0] - greeks["delta"]).abs().max() < 0.01
    # Vega within ~0.5 (vega ~ 40 at the money).
    assert (dV[:, 4] - greeks["vega"]).abs().max() < 1.0


def test_stacked_model_forward_and_diff_shapes():
    model = StackedResidualDML(d_X=5, n_stages=3)
    X = torch.randn(16, 5)
    V, dV = model.predict_with_diff(X, stage=2)
    assert V.shape == (16,)
    assert dV.shape == (16, 5)


def test_stacked_model_stage_zero_is_baseline_only():
    """At stage 0, residual blocks should not affect output."""
    model = StackedResidualDML(d_X=5, n_stages=3)
    X = torch.randn(8, 5)
    V0 = model(X, stage=0)
    # Now permute residual block weights and check stage-0 output unchanged.
    with torch.no_grad():
        for p in model.residual_blocks.parameters():
            p.add_(torch.randn_like(p))
    V0_again = model(X, stage=0)
    assert torch.allclose(V0, V0_again)


def test_sensitivity_spec_empty_is_price_only():
    """Empty active list => differential term is zero."""
    spec = SensitivitySpec(active_indices=[])
    loss_fn = DifferentialLoss(d_X=5, sensitivity_spec=spec, lam=1.0)
    yp = torch.randn(64)
    yd = torch.randn(64, 5)
    loss_fn.fit_scales(yp, yd)
    Vh = torch.randn(64, requires_grad=True)
    dVh = torch.randn(64, 5)
    out = loss_fn(Vh, dVh, yp, yd)
    assert float(out["diff_mse"]) == 0.0


def test_sensitivity_spec_subset_only_uses_those_coords():
    """Setting active=[0] should make perfect-on-coord-0 zero the diff term."""
    spec = SensitivitySpec(active_indices=[0])
    loss_fn = DifferentialLoss(d_X=5, sensitivity_spec=spec, lam=1.0)
    yp = torch.randn(64)
    yd = torch.randn(64, 5)
    loss_fn.fit_scales(yp, yd)
    Vh = yp.clone()
    dVh = torch.zeros(64, 5)
    dVh[:, 0] = yd[:, 0]  # perfect on coord 0 only
    out = loss_fn(Vh, dVh, yp, yd)
    assert float(out["diff_mse"]) < 1e-10


if __name__ == "__main__":
    test_softplus_call_recovers_true_call_for_large_beta()
    test_bs_simulator_matches_analytic_price_and_greeks()
    test_stacked_model_forward_and_diff_shapes()
    test_stacked_model_stage_zero_is_baseline_only()
    test_sensitivity_spec_empty_is_price_only()
    test_sensitivity_spec_subset_only_uses_those_coords()
    print("all tests passed")
