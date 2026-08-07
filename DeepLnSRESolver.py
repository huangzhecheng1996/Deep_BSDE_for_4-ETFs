import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import math

"""
DeepLnSRESolver: Deep BSDE solver for multi-factor logarithmic Stochastic Riccati Equations.

- Alternating Optimization: Supports independent freezing of P0 and Lambda networks to stabilize early-stage training.
- Gradient Clipping: Enforces strict norm bounds to prevent exploding gradients caused by quadratic drift terms and extreme variance paths.
- Dynamic LR Scheduling: Implements ReduceLROnPlateau with moving-average loss buffering for adaptive convergence.
"""


class DeepLnSRESolver(nn.Module):
    def __init__(self, params, lambda_net, sre_equation, dt=1/252, T=1.0, device='cpu',
                 initial_p0=0.0):
        super(DeepLnSRESolver, self).__init__()
        self.params = params
        self.dt = dt
        self.T = T
        self.N_steps = int(T / dt)
        self.device = device
        self.lambda_net = lambda_net.to(device)
        self.sre_equation = sre_equation.to(device)
        self.P0 = nn.Parameter(torch.tensor([initial_p0], dtype=torch.float64, device=device))
        self.current_epoch = 0
        self.history = {
            'epoch': [], 'loss': [], 'yt_mean': [], 'yt_std': [],
            'grad_norm_lambda': [], 'grad_norm_p0': [], 'p0': [],
            'learning_rate_lambda': [], 'learning_rate_p0': [], 'freeze_p0': [], 'freeze_lambda': []
        }
        self.optimizer_state = None
        self.scheduler_state = None


    def forward_simulation(self, V_paths, dW_paths):
        batch_size = V_paths.shape[0]
        check_P_t = self.P0.expand(batch_size, 1)
        check_Lambda_t = self.lambda_net(0, V_paths[:, 0, :])
        for i in range(self.N_steps):
            t_curr = (i * self.dt) / self.T
            V_curr = V_paths[:, i, :]
            dW_curr = dW_paths[:, i, :]

            drift = self.sre_equation(t_curr, check_P_t, check_Lambda_t, V_curr)
            diffusion_term = torch.sum(check_Lambda_t * dW_curr, dim=1, keepdim=True)
            check_P_t = check_P_t + drift * self.dt + diffusion_term
            V_next = V_paths[:, i + 1, :]
            if i + 1 < self.N_steps:
                check_Lambda_t = self.lambda_net(i + 1, V_next)
        return check_P_t

    def train_model(self, simulator, additional_epochs=1200,
                    lr_lambda=1e-3, lr_p0=5e-2, batch_size=1024,
                    freeze_p0=False, freeze_lambda=False,
                    reset_optim_state=True,patience=30):

        optimizer = optim.Adam([
            {'params': self.lambda_net.parameters(), 'lr': lr_lambda},
            {'params': [self.P0], 'lr': lr_p0}
        ])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
        start_epoch = self.current_epoch

        if reset_optim_state:
            print(f"[Info] Reset opt state")
        else:
            if self.optimizer_state is not None:
                optimizer.load_state_dict(self.optimizer_state)
                optimizer.param_groups[0]['lr'] = lr_lambda
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]['lr'] = lr_p0
                if self.scheduler_state is not None:
                    scheduler.load_state_dict(self.scheduler_state)
                    scheduler.num_bad_epochs = 0
                    scheduler.best = math.inf

        target_epoch = start_epoch + additional_epochs
        status = "[Fix P0]" if freeze_p0 else "[Fix Lam]" if freeze_lambda else "[Joint]"

        print("\n" + "=" * 135)
        print(f" Train | Mode: {status} | Add: {additional_epochs} | Target: {target_epoch}")
        print("-" * 135)
        print(f"{'Epoch':<7} | {'Train_Loss':<11} | {'Eval_Loss':<11} | {'P_T_Mean':<10} | {'P_T_Std':<9} | "
              f"{'Grad_Lam':<9} | {'Grad_P0':<9} | {'LR_Lam':<9} | {'LR_P0':<9} | {'check_P0':<10} | {'P0_Exp':<9}")
        print("-" * 135)

        try:
            loss_buffer = []

            for epoch in range(start_epoch, target_epoch):
                if os.path.exists("stop.txt"):
                    print(f"\n[Stop] stop.txt detected. Exit.")
                    os.remove("stop.txt")
                    break

                V_batch, dW_batch = simulator.simulate_tensors(
                    num_days=self.N_steps, dt=self.dt, num_scenarios=batch_size
                )
                V_batch = V_batch.detach()
                dW_batch = dW_batch.detach()

                self.lambda_net.train()
                optimizer.zero_grad()

                check_P_T_pred = self.forward_simulation(V_batch, dW_batch)
                check_P_T_mean = torch.mean(check_P_T_pred)
                check_P_T_var = torch.var(check_P_T_pred)

                loss = 1.0 * (check_P_T_mean - 0.0)**2 + check_P_T_var
                loss.backward()

                if freeze_p0:
                    if self.P0.grad is not None: self.P0.grad.zero_()
                if freeze_lambda:
                    for p in self.lambda_net.parameters():
                        if p.grad is not None: p.grad.zero_()

                if not freeze_lambda:
                    torch.nn.utils.clip_grad_norm_(self.lambda_net.parameters(), max_norm=3.0)
                if not freeze_p0:
                    torch.nn.utils.clip_grad_norm_([self.P0], max_norm=5.0)

                optimizer.step()

                epoch_loss = loss.item()

                loss_buffer.append(epoch_loss)
                if len(loss_buffer) > 10:
                    loss_buffer.pop(0)

                current_lr_lambda = optimizer.param_groups[0]['lr']
                current_lr_p0 = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0
                is_warming_up = len(loss_buffer) < 10

                if not is_warming_up:
                    old_lr_lambda = optimizer.param_groups[0]['lr']

                    smoothed_loss = sum(loss_buffer) / 10.0
                    scheduler.step(smoothed_loss)

                    new_lr_lambda = optimizer.param_groups[0]['lr']

                    if new_lr_lambda < old_lr_lambda:
                        scheduler.best = math.inf
                        scheduler.num_bad_epochs = 0

                    recorded_loss = smoothed_loss
                else:
                    recorded_loss = float('nan')

                if epoch % 10 == 0 or epoch == target_epoch - 1:
                    with torch.no_grad():
                        self.lambda_net.eval()
                        V_eval, dW_eval = simulator.simulate_tensors(
                            num_days=self.N_steps, dt=self.dt, num_scenarios=batch_size
                        )
                        V_eval = V_eval.detach()
                        dW_eval = dW_eval.detach()

                        check_P_T_batch = self.forward_simulation(V_eval, dW_eval)
                        eval_loss = torch.mean(check_P_T_batch ** 2).item()
                        mean_check_p = check_P_T_batch.mean().item()
                        std_check_p = check_P_T_batch.std().item()

                    current_check_p0 = self.P0.item()

                    grad_norm_lambda_sq = 0.0
                    for p in self.lambda_net.parameters():
                        if p.grad is not None:
                            grad_norm_lambda_sq += p.grad.norm().item() ** 2
                    grad_norm_lambda = grad_norm_lambda_sq ** 0.5

                    grad_norm_p0_sq = 0.0
                    if self.P0.grad is not None:
                        grad_norm_p0_sq += self.P0.grad.norm().item() ** 2
                    grad_norm_p0 = grad_norm_p0_sq ** 0.5

                    self.history['epoch'].append(epoch)
                    self.history['loss'].append(recorded_loss)
                    self.history['yt_mean'].append(mean_check_p)
                    self.history['yt_std'].append(std_check_p)
                    self.history['grad_norm_lambda'].append(grad_norm_lambda)
                    self.history['grad_norm_p0'].append(grad_norm_p0)
                    self.history['p0'].append(current_check_p0)
                    self.history['learning_rate_lambda'].append(current_lr_lambda)
                    self.history['learning_rate_p0'].append(current_lr_p0)
                    self.history['freeze_p0'].append(freeze_p0)
                    self.history['freeze_lambda'].append(freeze_lambda)

                    display_train_loss = f"{smoothed_loss:.2e}" if not is_warming_up else f"{epoch_loss:.2e}"

                    print(f"{epoch:<7d} | {display_train_loss:<11} | {eval_loss:<11.2e} | "
                          f"{mean_check_p:<+10.5f} | {std_check_p:<9.5f} | "
                          f"{grad_norm_lambda:<9.2e} | {grad_norm_p0:<9.2e} | "
                          f"{current_lr_lambda:<9.2e} | {current_lr_p0:<9.2e} | "
                          f"{current_check_p0:<+10.6f} | {math.exp(current_check_p0):<9.6f}")

                self.current_epoch = epoch + 1

        except KeyboardInterrupt:
            print(f"\n[Int] Interrupted. Epoch: {self.current_epoch}")

        self.optimizer_state = optimizer.state_dict()
        self.scheduler_state = scheduler.state_dict()

        print("-" * 135)
        print(f"========== Done (Epoch: {self.current_epoch}) ==========")
        self.export_history_to_csv()
        return self.P0.item()

    def export_history_to_csv(self, csv_path='sre_logspace_history.csv'):
        if not self.history['epoch']:
            return
        df = pd.DataFrame(self.history)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f">>> [CSV] Saved: {csv_path}")

    def save_model(self, file_path='sre_logspace_model.pt', csv_path='sre_logspace_history.csv'):
        save_dict = {
            'lambda_net_state': self.lambda_net.state_dict(),
            'P0': self.P0.item(),
            'env_params': self.params,
            'current_epoch': self.current_epoch,
            'history': self.history,
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(save_dict, file_path)
        print(f">>> [Model] Saved: {file_path}")
        self.export_history_to_csv(csv_path)

    def load_model(self, file_path='sre_logspace_model.pt'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Not found: {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        self.lambda_net.load_state_dict(checkpoint['lambda_net_state'])
        self.P0.data = torch.tensor([checkpoint['P0']], dtype=torch.float64, device=self.device)

        self.current_epoch = checkpoint.get('current_epoch', 0)
        loaded_history = checkpoint.get('history', {})
        epochs_len = len(loaded_history.get('epoch', []))

        self.history = {
            'epoch': list(loaded_history.get('epoch', [])),
            'loss': list(loaded_history.get('loss', [])),
            'yt_mean': list(loaded_history.get('yt_mean', [])),
            'yt_std': list(loaded_history.get('yt_std', [])),
            'grad_norm_lambda': list(loaded_history.get('grad_norm_lambda', [0.0] * epochs_len)),
            'grad_norm_p0': list(loaded_history.get('grad_norm_p0', [0.0] * epochs_len)),
            'p0': list(loaded_history.get('p0', [])),
            'learning_rate_lambda': list(loaded_history.get('learning_rate_lambda', [0.0] * epochs_len)),
            'learning_rate_p0': list(loaded_history.get('learning_rate_p0', [0.0] * epochs_len)),
            'freeze_p0': list(loaded_history.get('freeze_p0', [False] * epochs_len)),
            'freeze_lambda': list(loaded_history.get('freeze_lambda', [False] * epochs_len))
        }

        self.optimizer_state = checkpoint.get('optimizer_state', None)
        self.scheduler_state = checkpoint.get('scheduler_state', None)

        print(f">>> [Model] Loaded: {file_path}")
        print(f"    Epoch: {self.current_epoch} | check_P0: {self.P0.item():.6f} | P0_Exp: {math.exp(self.P0.item()):.6f}")
        return checkpoint.get('env_params', {})