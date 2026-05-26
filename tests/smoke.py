"""Smoke test: tiny version of the example to verify the full pipeline runs."""
import torch

from diff_boosting.simulators import BlackScholesSimulator
from diff_boosting.models import StackedResidualDML
from diff_boosting.training import (
    DifferentialBoostingTrainer, CurriculumSchedule, SensitivitySpec,
)
from diff_boosting.training.trainer import StageConfig

torch.manual_seed(0)
device = "cpu"
n = 500
g = torch.Generator(device=device).manual_seed(1)
S0 = 80.0 + 40.0 * torch.rand(n, generator=g)
K = 80.0 + 40.0 * torch.rand(n, generator=g)
T = 0.05 + 1.95 * torch.rand(n, generator=g)
r = 0.02 * torch.ones(n)
sigma = 0.10 + 0.30 * torch.rand(n, generator=g)
X_train = torch.stack([S0, K, T, r, sigma], dim=1)
X_val = X_train[:100]

sim = BlackScholesSimulator(antithetic=True)
model = StackedResidualDML(d_X=5, n_stages=2,
                           hidden_baseline=(32, 32),
                           hidden_residual=(16, 16))
spec = SensitivitySpec(active_indices=[0, 4], weights={4: 5.0})
trainer = DifferentialBoostingTrainer(model=model, simulator=sim,
                                      sensitivity_spec=spec, lam=1.0, device=device)
schedule = CurriculumSchedule(n_stages=2, width_init=5.0, width_final=0.1, p=2.0)
cfgs = [StageConfig(n_epochs=50, lr=1e-3, patience=20, log_every=25),
        StageConfig(n_epochs=50, lr=5e-4, patience=20, log_every=25)]
trainer.train_with_simulator(X_train=X_train, X_val=X_val, schedule=schedule,
                             stage_configs=cfgs, n_paths=1024, seed=42)
print("\nsmoke test passed")
