import torch
import numpy as np
import math

class DeepBSDE_Strategy_LogSpace:
    def __init__(self, initial_check_p0, q0, model_params, r_f, dt, T_horizon, lambda_net, sre_eq, device='cpu'):
        self.check_p0 = float(initial_check_p0) 
        self.p0 = math.exp(self.check_p0) 
        self.q0 = float(q0)
        self.params = model_params
        self.r_f = r_f
        self.dt = dt
        self.T = T_horizon
        self.device = torch.device(device)
        self.assets = ['xa', 'xb', 'xc', 'xd'] 
        
        self.lambda_net = lambda_net.to(self.device)
        self.lambda_net.eval()
        self.sre_eq = sre_eq.to(self.device)
        
        
        self._precompute_constants()

        self.current_check_P = torch.tensor([self.check_p0], dtype=torch.float64, device=self.device)
        
        self.last_time = None
        self.last_f = None
        self.last_check_Lambda = None
        self.last_sigma_pinv = None
        
        self._daily_log_cache = {}

        print("Init")

    def _precompute_constants(self):
        p = self.params
        m_wm_global = p.get('m_WM', 0.0)
        suffix_map = {'xa': 'a', 'xb': 'b', 'xc': 'c', 'xd': 'd'}
        
        def to_t(x): return torch.tensor(x, device=self.device, dtype=torch.float64)
        
        self.gamma, self.rho_M, self.delta, self.m_t, self.n_t = {}, {}, {}, {}, {}

        for k in self.assets:
            s = suffix_map[k]
            self.gamma[k] = to_t(p.get(f'gamma_{s}M', 0.0))
            self.rho_M[k] = to_t(p.get(f'rho_{s}M', 0.0))
            self.delta[k] = to_t(p.get(f'delta_{s}', 0.0)) 
            self.m_t[k] = to_t(p.get(f'm_W{s}', 0.0))
            
            vega_prem = p.get(f'm_Z{s}M', 0.0)
            delta_prem = p.get(f'delta_{s}', 0.0) * m_wm_global
            self.n_t[k] = to_t(vega_prem + delta_prem)

    def calculate_weights(self, snapshot, target_return: float, initial_capital: float, allow_leverage: bool, time_remain: float) -> dict:
        var_vals = snapshot.history_window.var().values * 252.0
        var_vals = np.nan_to_num(var_vals, nan=1e-4)
        vm_scalar = snapshot.vm_lagged
        
        vm_tensor = torch.tensor(vm_scalar, device=self.device, dtype=torch.float64)
        v_total = torch.tensor(var_vals, device=self.device, dtype=torch.float64)
        
        v_intrinsic = {}
        for i, k in enumerate(self.assets):
            factor = self.gamma[k]**2 + self.delta[k]**2
            v_sys = factor * vm_tensor
            v_raw = v_total[i] - v_sys
            v_intrinsic[k] = torch.maximum(v_raw, torch.tensor(1e-6, device=self.device, dtype=torch.float64))

        t_current = self.T - time_remain
        t_tensor = torch.tensor([[t_current]], dtype=torch.float64, device=self.device)

        step_idx = int(round(t_current / self.dt))
        step_idx = max(0, min(step_idx, self.lambda_net.num_steps - 1))

        V_t_list = [v_intrinsic[k].item() for k in self.assets]
        V_t_list.append(max(vm_scalar, 1e-6))
        V_t_tensor = torch.tensor([V_t_list], dtype=torch.float64, device=self.device)

        with torch.no_grad():
            check_Lambda_t = self.lambda_net(step_idx, V_t_tensor)
            
            _, _, sigma_t = self.sre_eq._build_tensors(V_t_tensor)
            sigma_t_sq = sigma_t.squeeze(0) if sigma_t.dim() == 3 else sigma_t

            if self.last_time is not None:
                dt_step = t_current - self.last_time
                if dt_step > 0:
                    residual = torch.tensor(snapshot.lagged_residual, dtype=torch.float64, device=self.device).unsqueeze(1)
                    dW_approx = torch.matmul(self.last_sigma_pinv, residual).squeeze(1)

                    diffusion_term = torch.matmul(self.last_check_Lambda.squeeze(0), dW_approx).unsqueeze(0)
                    diffusion_term = torch.clamp(diffusion_term, min=-0.3, max=0.3)
                    
                    drift_term = self.last_f * dt_step
                    delta_check_P = drift_term + diffusion_term
                    
                    self.current_check_P = (self.current_check_P + delta_check_P).view(-1)
                    self.current_check_P = torch.clamp(self.current_check_P, min=math.log(1e-6), max=0.0)

            check_P_tensor = self.current_check_P.view(-1, 1)
            f_t = self.sre_eq(t_tensor, check_P_tensor, check_Lambda_t, V_t_tensor)
            
            self.last_time = t_current
            self.last_check_Lambda = check_Lambda_t
            self.last_f = f_t
            self.last_sigma_pinv = torch.pinverse(sigma_t_sq)

            proj_check_Lambda = torch.bmm(sigma_t if sigma_t.dim() == 3 else sigma_t.unsqueeze(0), check_Lambda_t.unsqueeze(-1)).squeeze(-1).view(-1)
             
            lambda_b_tensor = proj_check_Lambda
            lambda_b_tensor = torch.clamp(lambda_b_tensor, min=-50.0, max=50.0)
            lambda_b_dict = {k: lambda_b_tensor[i].item() for i, k in enumerate(self.assets)}

        g_myopic = self._compute_g(v_intrinsic, vm_scalar)
        g_total = self._compute_g(v_intrinsic, vm_scalar, extra_b=lambda_b_dict)
        
        P_0 = self.p0 
        h0 = self.q0   
        x0 = initial_capital
        d_tgt = initial_capital * np.exp(target_return * self.T)
        
        denom = 1.0 - P_0 * (h0**2)
        c_star = (P_0 * h0 * x0 - d_tgt) / denom if abs(denom) > 1e-9 else 0.0
        
        q_t = np.exp(-self.r_f * time_remain)
        equity = max(snapshot.total_equity, 1e-4)
        
        feedback = snapshot.total_equity + c_star * q_t
        
        w_raw, w_myopic = [], []
        for k in self.assets:
            u_tot = -1.0 * g_total[k].item() * feedback
            u_myo = -1.0 * g_myopic[k].item() * feedback
            w_raw.append(u_tot / equity)
            w_myopic.append(u_myo / equity)
            
        w_final = np.array(w_raw)

        if not allow_leverage:
            s = np.sum(np.abs(w_final))
            if s > 1.0: w_final /= s

        return dict(zip(self.assets, w_final))

    def _compute_g(self, V_est, vm_scalar, extra_b=None): 
        V = {k: torch.clamp(v, min=1e-9) for k, v in V_est.items()}
        V['M'] = torch.maximum(torch.tensor(vm_scalar, device=self.device, dtype=torch.float64), 
                               torch.tensor(1e-6, device=self.device, dtype=torch.float64))
        sqrtV_M = torch.sqrt(V['M'])
        
        rho_M_own_val = self.params.get('rho_M_own', 0.0)
        rho_M_own = torch.tensor(rho_M_own_val, device=self.device, dtype=torch.float64)
        
        D_inv, v_vec, z_vec = {}, {}, {}
        for k in self.assets:
            d = V[k] + self.gamma[k]**2 * (1 - self.rho_M[k]**2) * V['M']
            D_inv[k] = 1.0 / d
            
            v_vec[k] = (self.gamma[k] * self.rho_M[k] + self.delta[k] * rho_M_own) * sqrtV_M
            z_vec[k] = self.delta[k] * torch.sqrt(1.0 - rho_M_own**2) * sqrtV_M
            
        S_vv = sum(v_vec[k]**2 * D_inv[k] for k in self.assets)
        S_zz = sum(z_vec[k]**2 * D_inv[k] for k in self.assets)
        S_vz = sum(v_vec[k] * z_vec[k] * D_inv[k] for k in self.assets)
        
        detM = torch.clamp((1 + S_vv) * (1 + S_zz) - S_vz**2, min=1e-9)
        lam_11, lam_22, lam_12 = (1 + S_zz)/detM, (1 + S_vv)/detM, -S_vz/detM
        
        b_vec = {k: self.m_t[k]*V[k] + self.n_t[k]*V['M'] for k in self.assets}
        
        if extra_b is not None:
            for k in self.assets:
                b_vec[k] = b_vec[k] + extra_b[k]
                
        h_v = sum(b_vec[k]*v_vec[k]*D_inv[k] for k in self.assets)
        h_z = sum(b_vec[k]*z_vec[k]*D_inv[k] for k in self.assets)
        
        w_v = h_v * lam_11 + h_z * lam_12
        w_z = h_v * lam_12 + h_z * lam_22
        
        g = {}
        for k in self.assets:
            corr = w_v * v_vec[k] + w_z * z_vec[k]
            g[k] = (b_vec[k] - corr) * D_inv[k]
        return g