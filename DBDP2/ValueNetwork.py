import torch
import torch.nn as nn

class SubNetwork(nn.Module):
    """
    Feed-forward network for a single time step in the DBDP2 scheme.
    Uses Tanh activation to ensure smooth, bounded gradients for autograd differentiation.
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers).to(torch.float64)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    
    """
    Global value network estimating the LnSRE solution across all discretization steps.
    """
    
    def __init__(self, num_steps, state_dim=5, hidden_layers=None):
        super().__init__()
        self.num_steps = num_steps
        self.state_dim = state_dim
        input_dim = state_dim + state_dim
        self.sub_networks = nn.ModuleList([
            SubNetwork(input_dim, 1, hidden_layers) for _ in range(num_steps)
        ])

    def forward(self, step_idx, V_t):

        V_safe = torch.clamp(V_t, min=1e-8)
        sqrt_V = torch.sqrt(V_safe)
        x = torch.cat([V_t, sqrt_V], dim=1)
        check_P_pred = self.sub_networks[step_idx](x)

        return check_P_pred