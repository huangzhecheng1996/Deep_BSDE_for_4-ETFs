import torch
import torch.nn as nn

class lnSRE_Equation(nn.Module):
    """
    Drift of the logarithmic Stochastic Riccati Equation (lnSRE)
    for mean-variance portfolio under multi-factor SV.
    """
    def __init__(self, params, device='cpu'):
        super().__init__()
        self.params = params
        self.device = device
        self.assets = ['a', 'b', 'c', 'd']
        self.num_assets = len(self.assets)
        self.lambda_dim = 14
        self._precompute_tensors()

    def _precompute_tensors(self):
        def to_t(val):
            return torch.tensor(val, device=self.device, dtype=torch.float64)
        self.r = to_t(self.params.get('r', 0.0))
        self.rho_M_own = to_t(self.params.get('rho_M_own', 0.0))
        self.delta = torch.stack([to_t(self.params.get(f'delta_{a}', 0.0)) for a in self.assets])
        self.gamma = torch.stack([to_t(self.params.get(f'gamma_{a}M', 0.0)) for a in self.assets])
        self.rho_aM = torch.stack([to_t(self.params.get(f'rho_{a}M', 0.0)) for a in self.assets])
        self.rho_own = torch.stack([to_t(self.params.get(f'rho_{a}_own', 0.0)) for a in self.assets])
        self.m_W = torch.stack([to_t(self.params.get(f'm_W{a}', 0.0)) for a in self.assets])
        self.ell = torch.stack([to_t(self.params.get(f'ell_{a}', 0.0)) for a in self.assets])
        self.m_ZaM = torch.stack([to_t(self.params.get(f'm_Z{a}M', 0.0)) for a in self.assets])
        self.m_WM = to_t(self.params.get('m_WM', 0.0))
        self.ell_ZM = to_t(self.params.get('ell_ZM', 0.0))

        
    def _build_tensors(self, V_t):
        """
        Construct risk-premium vector B_t and regularised covariance Sigma_t.
        B_t follows the affine form: B_k = m_k * V_k + n_k * V_0
        Sigma_t = sigma * sigma^*
        """
        B = V_t.shape[0]
        eps = 1e-9
        V_assets = torch.clamp(V_t[:, :4], min=eps)
        V_M = torch.clamp(V_t[:, 4], min=eps)
        sqrt_V_assets = torch.sqrt(V_assets)
        sqrt_V_M = torch.sqrt(V_M).unsqueeze(-1)

        sigma_t = torch.zeros(B, self.num_assets, self.lambda_dim, dtype=torch.float64, device=self.device)
        for i in range(self.num_assets):
            sigma_t[:, i, i] = self.rho_own[i] * sqrt_V_assets[:, i]
            sigma_t[:, i, 4] = (self.gamma[i] * self.rho_aM[i] + self.delta[i] * self.rho_M_own) * sqrt_V_M.squeeze(-1)
            sigma_t[:, i, 5+i] = torch.sqrt(torch.clamp(1.0 - self.rho_own[i]**2, min=0.0)) * sqrt_V_assets[:, i]
            sigma_t[:, i, 9+i] = self.gamma[i] * torch.sqrt(torch.clamp(1.0 - self.rho_aM[i]**2, min=0.0)) * sqrt_V_M.squeeze(-1)
            sigma_t[:, i, 13] = self.delta[i] * torch.sqrt(torch.clamp(1.0 - self.rho_M_own**2, min=0.0)) * sqrt_V_M.squeeze(-1)

        Sigma_t = torch.bmm(sigma_t, sigma_t.transpose(1, 2))
        eye_4 = torch.eye(4, dtype=torch.float64, device=self.device).view(1, 4, 4)
        Sigma_t_reg = Sigma_t + eye_4 * 1e-12

        m = self.rho_own * self.ell + torch.sqrt(torch.clamp(1.0 - self.rho_own**2, min=0.0)) * self.m_W
        n = self.gamma * (self.rho_aM * self.ell_ZM + torch.sqrt(torch.clamp(1.0 - self.rho_aM**2, min=0.0)) * self.m_ZaM) + self.delta * self.m_WM

        B_t = m * V_assets + n * V_M.unsqueeze(-1)

        return B_t, Sigma_t_reg, sigma_t

        
    def forward(self, t, check_P_t, check_Lambda_t, V_t):
        """
        lnSRE drift:
            - [ (2r - |rho|^2) - 2<rho, check_Lambda> - check_Lambda^* sigma^* (sigma sigma^*)^{-1} sigma check_Lambda + 0.5|check_Lambda|^2 ]
        """
        B_t, Sigma_t, sigma_t = self._build_tensors(V_t)
        B_unsqueeze = B_t.unsqueeze(-1)
        Lambda_unsqueeze = check_Lambda_t.unsqueeze(-1)

        invSigma_B = torch.linalg.solve(Sigma_t, B_unsqueeze)
        rho_sq = torch.bmm(B_unsqueeze.transpose(1, 2), invSigma_B).squeeze(-1).squeeze(-1)

        proj_Lambda = torch.bmm(sigma_t, Lambda_unsqueeze)
        invSigma_proj = torch.linalg.solve(Sigma_t, proj_Lambda)
        cross_term = 2.0 * torch.bmm(B_unsqueeze.transpose(1, 2), invSigma_proj).squeeze(-1).squeeze(-1)
        proj_term = torch.bmm(proj_Lambda.transpose(1, 2), invSigma_proj).squeeze(-1).squeeze(-1)
        lambda_sq = torch.bmm(Lambda_unsqueeze.transpose(1, 2), Lambda_unsqueeze).squeeze(-1).squeeze(-1)

        bracket = (2.0 * self.r - rho_sq) - cross_term - proj_term + 0.5 * lambda_sq
        drift = -bracket

        return drift.unsqueeze(-1)