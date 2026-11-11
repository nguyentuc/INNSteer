"""
inn.py — Affine Coupling INN for Activation Space Steering
Architecture: RealNVP-style stacked coupling layers (exact invertibility)
"""

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """
    Affine coupling layer.
    Args:
        dim        : total feature dimension
        hidden_dim : MLP hidden width for s_net and t_net
        mask_type  : 'top' or 'bottom' — which half is the passthrough
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        mask_type: str = "top",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        self.rest = dim - self.half      # handles odd dims safely
        self.mask_type = mask_type

        in_features = self.half if mask_type == "top" else self.rest
        out_features = self.rest if mask_type == "top" else self.half

        # ── Scale network ─────────────────────────────────────────────────
        # Tanh bounds output => exp(s) stays in (exp(-1), exp(1)) ≈ (0.37, 2.72),
        # preventing the Jacobian diagonal from exploding.
        # Dropout is placed after each hidden ReLU, NOT after the final Tanh,
        # so the bounded scale property is always preserved.
        self.s_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  #  hidden dropout 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),   #  hidden dropout
            nn.Linear(hidden_dim, out_features),
            nn.Tanh(),     # output layer — no dropout here
        )

        # ── Translation network ───────────────────────────────────────────
        # Unbounded output. Dropout after each hidden ReLU only;
        # the final Linear is never dropped so t(x) is always a complete vector.
        self.t_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  #  hidden dropout (new)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout), #  hidden dropout (new)
            nn.Linear(hidden_dim, out_features), # output layer — no dropout here
        )

        # ── Near-identity initialisation ─────────────────────────────────
        # Zero the weights and biases of the final output layers so the untrained
        # INN starts as the identity map
        nn.init.zeros_(self.s_net[-2].weight) # Linear before Tanh
        nn.init.zeros_(self.s_net[-2].bias)
        nn.init.zeros_(self.t_net[-1].weight) # final Linear
        nn.init.zeros_(self.t_net[-1].bias)

    # ──────────────────────────────────────────────────────────────────────
    def _split(self, x):
        """Split x into (passthrough, active) halves."""
        if self.mask_type == "top":
            return x[:, : self.half], x[:, self.half :]
        else:
            return x[:, self.half :], x[:, : self.half]

    def _merge(self, pass_half, act_half):
        """Reconstruct full vector from the two halves."""
        if self.mask_type == "top":
            return torch.cat([pass_half, act_half], dim=1)
        else:
            return torch.cat([act_half, pass_half], dim=1)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        phi direction.  Returns (y, log_det_jacobian) — shape (B,d) and (B,).
        In training mode, dropout is active in s_net and t_net hidden layers.
        In eval mode (inn.eval()), dropout is disabled — exact invertibility holds.
        """
        x_pass, x_act = self._split(x)
        s = self.s_net(x_pass)              # (B, out_features)
        t = self.t_net(x_pass)
        y_act   = x_act * torch.exp(s) + t
        log_det = s.sum(dim=1)              # triangular Jacobian => sum of diagonal
        return self._merge(x_pass, y_act), log_det

    def inverse(self, y):
        """
        phi^-1 direction.  Exact, same compute cost as forward
        """
        y_pass, y_act = self._split(y)
        s = self.s_net(y_pass)
        t = self.t_net(y_pass)
        x_act = (y_act - t) * torch.exp(-s)
        return self._merge(y_pass, x_act)


# ──────────────────────────────────────────────────────────────────────────────

class INN(nn.Module):
    """
    Stacked affine coupling INN.
    phi(x)    => z  (activation space  => phi-space)
    phi_inv(z)  => x  (phi-space => activation space, exact in eval mode)
    Alternating mask_type ensures all dimensions are transformed across layers.
    Args:
        dim        : feature dimension (must match LLM hidden size)
        num_layers : number of stacked coupling layers
        hidden_dim : hidden width of each coupling MLP
        dropout    : dropout probability in coupling MLP hidden layers.
                     Passed through to every CouplingLayer.
                     0.0 = backward-compatible (no dropout).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.coupling_layers = nn.ModuleList(
            [
                CouplingLayer(
                    dim = dim,
                    hidden_dim = hidden_dim,
                    mask_type = "top" if i % 2 == 0 else "bottom",
                    dropout = dropout,      
                )
                for i in range(num_layers)
            ]
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """phi direction.  Returns (z, total_log_det)."""
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.coupling_layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z):
        """phi^-1 direction.  Runs coupling layers in reverse order."""
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
        return x

    # ──────────────────────────────────────────────────────────────────────
    # Convenience aliases used throughout the codebase
    def phi(self, x):
        """Forward pass, discarding log_det.  Always deterministic in eval mode."""
        z, _ = self.forward(x)
        return z

    def phi_inv(self, z):
        """Exact inverse.  Call inn.eval() first to disable dropout."""
        return self.inverse(z)