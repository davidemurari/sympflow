import torch
import torch.nn as nn
from torch.func import jacrev, vmap


class HamiltonianNet(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
        dtype: torch.dtype = torch.float32,
        d: int = 2,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.act = nn.Tanh()

        # Keep legacy attribute names for checkpoint compatibility.
        self.lift_U = nn.Linear(self.d // 2, hidden, dtype=dtype)
        self.proj_U = nn.Linear(hidden, 1, dtype=dtype)
        self.lift_K = nn.Linear(self.d // 2, hidden, dtype=dtype)
        self.proj_K = nn.Linear(hidden, 1, dtype=dtype)

        if self.n_layers > 2:
            self.linears_U = nn.ModuleList(
                [nn.Linear(hidden, hidden, dtype=dtype) for _ in range(self.n_layers - 2)]
            )
            self.linears_K = nn.ModuleList(
                [nn.Linear(hidden, hidden, dtype=dtype) for _ in range(self.n_layers - 2)]
            )

    def K(self, x: torch.Tensor) -> torch.Tensor:
        lifted = self.act(self.lift_K(x[:, self.d // 2 :]))
        if self.n_layers > 2:
            for i in range(self.n_layers - 2):
                lifted = self.act(self.linears_K[i](lifted))
        return self.proj_K(lifted)

    def U(self, x: torch.Tensor) -> torch.Tensor:
        lifted = self.act(self.lift_U(x[:, : self.d // 2]))
        if self.n_layers > 2:
            for i in range(self.n_layers - 2):
                lifted = self.act(self.linears_U[i](lifted))
        return self.proj_U(lifted)

    # Convenience aliases.
    def k(self, x: torch.Tensor) -> torch.Tensor:
        return self.K(x)

    def u(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(x).squeeze(-1) + self.K(x).squeeze(-1)


class SRNN(nn.Module):
    def __init__(
        self,
        hidden: int = 10,
        d: int = 2,
        dtype: torch.dtype = torch.float32,
        n_layers: int = 2,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.H = HamiltonianNet(hidden=hidden, dtype=dtype, d=d, n_layers=n_layers)
        self.d = d
        self.dt = dt

    @property
    def hamiltonian(self) -> HamiltonianNet:
        return self.H

    def grad_h(self, x: torch.Tensor) -> torch.Tensor:
        def h_scalar(y: torch.Tensor) -> torch.Tensor:
            return self.H(y.unsqueeze(0)).squeeze()

        return vmap(jacrev(h_scalar))(x)

    def symplectic_step(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        grad_h = self.grad_h(x)
        dpdt = -grad_h[:, : self.d // 2]
        p_new = x[:, self.d // 2 :] + dt * dpdt

        x_mid = torch.cat((x[:, : self.d // 2], p_new), dim=1)
        grad_h_mid = self.grad_h(x_mid)
        dqdt = grad_h_mid[:, self.d // 2 :]
        q_new = x_mid[:, : self.d // 2] + dt * dqdt

        return torch.cat([q_new, p_new], dim=1)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
        return self.symplectic_step(x, dt)
