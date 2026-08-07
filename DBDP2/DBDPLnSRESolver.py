import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import pandas as pd

class DBDPLnSRESolver(nn.Module):

    """
    Based on the Deep Backward Dynamic Programming (DBDP2) framework.
    """
    
    def __init__(self, params, value_net, sre_equation, dt=1/252, T=1.0, device='cpu'):
        super(DBDPLnSRESolver, self).__init__()
        self.params = params
        self.dt = dt
        self.T = T
        self.N_steps = int(T / dt)
        self.device = device

        self.value_net = value_net.to(device)
        self.sre_equation = sre_equation.to(device)

        self.P0 = nn.Parameter(torch.tensor([0.0], dtype=torch.float64, device=device))
        self.Lambda_active_0 = nn.Parameter(torch.zeros(1, 5, dtype=torch.float64, device=device))

        self.step_loss_history = {}

        self.sigma_V_tensor = torch.tensor([
            self.params.get('sigma_a'), self.params.get('sigma_b'),
            self.params.get('sigma_c'), self.params.get('sigma_d'),
            self.params.get('sigma_M')
        ], dtype=torch.float64, device=self.device)

        print(f"\n[Check] Init solver:")
        print(f"    -> N_steps: {self.N_steps}")

    @property
    def Lambda0(self):
        pad_zeros = torch.zeros(1, 9, dtype=torch.float64, device=self.device)
        return torch.cat([self.Lambda_active_0, pad_zeros], dim=1).detach()

    def _save_checkpoint(self, completed_steps, is_emergency=False, is_final=False):
        if is_final:
            filename = 'value_net_final_complete.pt'
            status_tag = "[Final]"
        elif is_emergency:
            filename = f'value_net_emergency_completed_{completed_steps}_steps.pt'
            status_tag = "[Emergency]"
        else:
            filename = f'value_net_ckpt_completed_{completed_steps}_steps.pt'
            status_tag = "[Stage]"

        save_dict = {
            'value_net_state': self.value_net.state_dict(),
            'completed_steps': completed_steps,
            'step_loss_history': self.step_loss_history,
            'P0': self.P0.item(),
            'Lambda_active_0': self.Lambda_active_0.detach().cpu(),
            'env_params': self.params
        }
        torch.save(save_dict, filename)
        print(f"  >>> {status_tag} Done: {completed_steps}/{self.N_steps} | File: {filename}")

    def _load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        completed_steps = checkpoint['completed_steps']

        interrupted_step = self.N_steps - completed_steps - 1
        state_dict = checkpoint['value_net_state']

        if interrupted_step >= 0:
            prefix = f'sub_networks.{interrupted_step}.'
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}

        self.value_net.load_state_dict(state_dict, strict=False)
        self.step_loss_history = checkpoint.get('step_loss_history', {})

        self.P0.data = torch.tensor([checkpoint.get('P0', 0.0)], dtype=torch.float64, device=self.device)
        if 'Lambda_active_0' in checkpoint:
            self.Lambda_active_0.data = checkpoint['Lambda_active_0'].to(self.device)

        if 'env_params' in checkpoint:
            self.params = checkpoint['env_params']

        return completed_steps

    def train_backward(self, simulator,
                       target_mse=1e-7,
                       max_epochs_list=None,
                       lr_start=1e-4,
                       lr_patience=50,
                       min_lr=5e-6,
                       batch_size=32768,
                       save_interval=50,
                       resume_ckpt_path=None,
                       global_warm_start=True):

        start_step = self.N_steps - 1
        completed_steps = 0
        terminal_val = self.params.get('target_terminal_ln_P', 0.0)

        if resume_ckpt_path is not None:
            print(f"\n[Resume] Loading from {resume_ckpt_path}...")
            loaded_completed_steps = self._load_checkpoint(resume_ckpt_path)
            start_step = self.N_steps - loaded_completed_steps - 1
            completed_steps = loaded_completed_steps
            print(f"[Resume] Skip {completed_steps} steps, start at i={start_step}.")

        if max_epochs_list is None:
            max_epochs_list = [400 if i < 30 else 100 for i in range(self.N_steps)]

        start_mode_tag = "Global Warm Start / Fine-Tuning" if global_warm_start else "Cold Start"
        print(f"\n========== Start | Target: {target_mse:.2e} | {start_mode_tag} ==========")

        try:
            for i in range(start_step, -1, -1):
                t_curr = (i * self.dt) / self.T
                curr_max_epochs = max_epochs_list[i]

                current_net = self.value_net.sub_networks[i]

                if i < self.N_steps - 1 and (not global_warm_start or i == 0):
                    trained_prev_net = self.value_net.sub_networks[i + 1]
                    current_net.load_state_dict(trained_prev_net.state_dict())

                optimizer = optim.Adam(current_net.parameters(), lr=lr_start)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)
                current_net.train()

                final_epoch_loss = float('inf')
                recent_losses = []

                for epoch in range(curr_max_epochs):

                    V_paths, dW_paths = simulator.simulate_tensors(
                        num_days=i + 1, dt=self.dt, num_scenarios=batch_size
                    )
                    V_paths = V_paths.detach()
                    dW_paths = dW_paths.detach()

                    V_batch = V_paths[:, i, :].clone().requires_grad_(True)
                    dW_batch = dW_paths[:, i, :]

                    if i == self.N_steps - 1:
                        target_batch = torch.full((batch_size, 1), terminal_val, dtype=torch.float64, device=self.device)
                    else:
                        with torch.no_grad():
                            self.value_net.sub_networks[i + 1].eval()
                            target_batch = self.value_net(i + 1, V_paths[:, i + 1, :]).detach()

                    optimizer.zero_grad()

                    check_P_i = self.value_net(i, V_batch)

                    grad_check_P = torch.autograd.grad(
                        outputs=check_P_i,
                        inputs=V_batch,
                        grad_outputs=torch.ones_like(check_P_i),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    Lambda_active = grad_check_P * self.sigma_V_tensor * torch.sqrt(torch.clamp(V_batch, min=1e-8))

                    pad_zeros = torch.zeros(batch_size, 9, dtype=torch.float64, device=self.device)
                    check_Lambda_i = torch.cat([Lambda_active, pad_zeros], dim=1)

                    drift = self.sre_equation(t_curr, check_P_i, check_Lambda_i, V_batch)
                    diffusion = torch.sum(check_Lambda_i * dW_batch, dim=1, keepdim=True)

                    check_P_next_pred = check_P_i + drift * self.dt + diffusion
                    loss = torch.mean((check_P_next_pred - target_batch) ** 2)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(current_net.parameters(), max_norm=3.0)

                    optimizer.step()
                    final_epoch_loss = loss.item()

                    recent_losses.append(final_epoch_loss)
                    if len(recent_losses) > 10:
                        recent_losses.pop(0)
                    smooth_mse = sum(recent_losses) / len(recent_losses)

                    if len(recent_losses) == 10:
                        scheduler.step(smooth_mse)

                    if len(recent_losses) == 10 and smooth_mse <= target_mse:
                        break

                    if (i % 10 == 0 or i == self.N_steps - 1 or i == 0) and (epoch % 50 == 0 or epoch == curr_max_epochs - 1):
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"  [Value Step {i:3d}] Epoch {epoch:3d}/{curr_max_epochs} | LR: {current_lr:.2e} | MSE Loss: {final_epoch_loss:.4e} | 10-Avg: {smooth_mse:.4e}")

                self.step_loss_history[i] = smooth_mse if len(recent_losses) == 10 else final_epoch_loss

                if i == 0:
                    self.P0.data = torch.tensor([check_P_i.mean().item()], dtype=torch.float64, device=self.device)
                    self.Lambda_active_0.data = Lambda_active[0:1, :].detach().clone()

                final_lr = optimizer.param_groups[0]['lr']
                is_early_stopped = (len(recent_losses) == 10 and smooth_mse <= target_mse)
                stop_reason = "Early stop" if is_early_stopped else "Max epoch"
                final_display_mse = smooth_mse if len(recent_losses) == 10 else final_epoch_loss
                print(f"  >>> [Step {i:3d} Done - {stop_reason}] Epochs: {epoch+1}/{curr_max_epochs} | LR: {final_lr:.2e} | MSE: {final_display_mse:.4e}")

                if i % 10 == 0 or i == 1 or i == 0:
                    if i == 0:
                        current_mean_lnP = self.P0.item()
                        print(f"  => Step   0 | \\check{{P}}_0 = {current_mean_lnP:.6f} (P = {math.exp(current_mean_lnP):.6f})")
                        print(f"     [Result] Norm Lambda_0: {torch.norm(self.Lambda0).item():.6f}\n")
                    else:
                        current_mean_lnP = check_P_i.mean().item()
                        print(f"  => Step {i:3d} | Mean \\check{{P}}_{i} = {current_mean_lnP:.6f} (P = {math.exp(current_mean_lnP):.6f})\n")

                completed_steps = self.N_steps - i
                if completed_steps % save_interval == 0 and i != 0:
                    self._save_checkpoint(completed_steps)

        except KeyboardInterrupt:
            print("\n[Interrupt] KeyboardInterrupt! Saving...")
            self._save_checkpoint(completed_steps, is_emergency=True)
            print("  >>> [System] Stopped safely.")
            return self.P0.item()

        print("========== All done! ==========")
        self._save_checkpoint(self.N_steps, is_final=True)

        return self.P0.item()

    def save_model(self, file_path='sre_dbdp2_value_model.pt'):
        save_dict = {
            'value_net_state': self.value_net.state_dict(),
            'env_params': self.params,
            'step_loss_history': self.step_loss_history,
            'P0': self.P0.item(),
            'Lambda_active_0': self.Lambda_active_0.detach().cpu()
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(save_dict, file_path)
        print(f"\n>>> [Model] Saved!")
        print(f"    - P0={self.P0.item():.4f}")
        print(f"    Path: {file_path}")

    def load_model(self, file_path='sre_dbdp2_value_model.pt'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Not found: {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)

        self.value_net.load_state_dict(checkpoint['value_net_state'])
        self.step_loss_history = checkpoint.get('step_loss_history', {})

        self.P0.data = torch.tensor([checkpoint.get('P0', 0.0)], dtype=torch.float64, device=self.device)
        if 'Lambda_active_0' in checkpoint:
            self.Lambda_active_0.data = checkpoint['Lambda_active_0'].to(self.device)

        return checkpoint.get('env_params', self.params)