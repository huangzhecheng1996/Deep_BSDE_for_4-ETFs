import torch
from torch.distributions import Gamma, Poisson
import pandas as pd
import numpy as np
import math
import warnings

class StochasticVolatilityModel:
    """
    Multi-factor CIR variance path generator.
    Discretisation: Full Truncation Euler (Lord, Koekkoek, Van Dijk, 2010).
    """
    def __init__(self, params):
        self.params = params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def _to_tensor(self, value):
        return torch.tensor(value, dtype=self.dtype, device=self.device) if not isinstance(value, torch.Tensor) else value.to(self.device).to(self.dtype)

    def simulate_tensors(self, num_days, dt=1/252, seed=None, num_scenarios=1000):
        """
        Full Truncation scheme:
            V_tilde_{t+1} = V_tilde_t + (alpha - beta * V_tilde_t^+) * dt
                            + sigma * sqrt(V_tilde_t^+) * dZ
            V_{t+1}       = max(V_tilde_{t+1}, eps)

        """
        if seed is not None:
            torch.manual_seed(seed + 50)

        B = num_scenarios
        N = num_days
        sq_dt = math.sqrt(dt)
        eps = 1e-12

        var_names = ['a', 'b', 'c', 'd', 'M']
        alpha = torch.stack([self._to_tensor(self.params[f'alpha_{v}']) for v in var_names])
        beta  = torch.stack([self._to_tensor(self.params[f'beta_{v}'])  for v in var_names])
        sigma = torch.stack([self._to_tensor(self.params[f'sigma_{v}']) for v in var_names])
        V_0   = torch.stack([self._to_tensor(self.params[f'V_{v}_0'])   for v in var_names])

        V_tensor  = torch.zeros(B, N + 1, 5, dtype=self.dtype, device=self.device)
        dW_tensor = torch.zeros(B, N, 14, dtype=self.dtype, device=self.device)

        V_tensor[:, 0, :] = V_0
        V_tilde = V_0.clone().unsqueeze(0).repeat(B, 1)  

        for t in range(N):
            dW = torch.randn(B, 14, dtype=self.dtype, device=self.device) * sq_dt
            dW_tensor[:, t, :] = dW
            dZ = dW[:, :5]

            V_safe = torch.clamp(V_tilde, min=0.0)       
            drift = (alpha - beta * V_safe) * dt
            diffusion = sigma * torch.sqrt(V_safe) * dZ
            V_tilde = V_tilde + drift + diffusion         
            V_tensor[:, t + 1, :] = torch.clamp(V_tilde, min=eps)

        return V_tensor, dW_tensor