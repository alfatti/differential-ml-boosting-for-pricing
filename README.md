# Differential Boosting

Stacked-residual differential machine learning for option pricing. Combines the
residual/boosting structure from Eshkofti & Barreau's vanishing-stacked-residual
PINN with Huge & Savine's differential ML.

## Idea in one paragraph

A vanilla DML net struggles on European calls particularly in the ITM region
because pathwise differentials carry little information there (delta saturates,
gamma and vega collapse). We fix this by training a sequence of networks
boosting-style: a smooth baseline first, then residual correction blocks that
each target a sharper version of the payoff. The "sharpness curriculum" is the
analogue of vanishing viscosity in the PINN paper.

## Architecture

```
Stage 0:    V^(0) = MLP_0(X)
Stage i>0:  V^(i) = V^(i-1) + |alpha_i| * MLP_i([X, V^(i-1)])
```

Each stage `i` is trained on labels generated with payoff smoothing
`g_{beta_i}(S,K) = (1/beta_i) * softplus(beta_i*(S-K))`. The schedule starts
with a large smoothing width and shrinks toward (but does not reach) the true
kink, so the final stage is a tight pricer rather than an unrepresentable
kinked function.

## Sensitivity control

You pick which AAD coordinates enter the loss:

```python
from diff_boosting.training import SensitivitySpec

# all sensitivities (Huge-Savine default)
spec = SensitivitySpec()

# delta and vega only, vega upweighted 5x
spec = SensitivitySpec(active_indices=[0, 4], weights={4: 5.0})

# price-only
spec = SensitivitySpec(active_indices=[])
```

## Model-agnostic

The simulator is the only model-specific component. To add Heston / SABR / etc.,
implement a subclass of `Simulator` whose `simulate(X, payoff_fn, n_paths)`
returns `(V, dV/dX)` via `torch.autograd`. Everything else (architecture,
losses, trainer, curriculum) is unchanged.

## Run

```bash
pip install -e .
python examples/european_call_bs.py
python -m pytest tests/ -v
```

## Layout

```
diff_boosting/
  simulators/   # base.py, payoffs.py, black_scholes.py
  models/       # residual_dml.py
  training/     # losses.py, trainer.py
  evaluation/   # metrics.py (BS oracle + vega-bps)
examples/european_call_bs.py
tests/test_pipeline.py
```

## Reference

K. Eshkofti and M. Barreau, *Vanishing Stacked-Residual PINN for State
Reconstruction of Hyperbolic Systems*, arXiv:2503.14222, 2025.

B. Huge and A. Savine, *Differential Machine Learning*, 2020.
