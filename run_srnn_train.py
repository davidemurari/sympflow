import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.model_quality_core import get_device, get_system_parameters, get_vector_field
from scripts.srnn_model import SRNN


def parse_int_list(values: str) -> list[int]:
    """Parse a comma-separated string into integers.

    Inputs: `values` (str) with items like "1,2,3".
    Returns: list[int].
    """
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_list(values: str) -> list[float]:
    """Parse a comma-separated string into floats.

    Inputs: `values` (str) with items like "0.1,0.2".
    Returns: list[float].
    """
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def format_eps(epsilon: float) -> str:
    """Format epsilon for checkpoint filenames.

    Inputs: `epsilon` (float).
    Returns: normalized string representation of epsilon.
    """
    return str(float(epsilon))


def generate_data_supervised_srnn(
    vec,
    system_parameters: dict,
    dtype: torch.dtype,
    n_samples: int,
    n_times: int,
    epsilon: float,
    dt: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """Generate SRNN supervised trajectories with optional additive noise.

    Inputs: vector field `vec`, `system_parameters`, tensor `dtype`, number of
    samples `n_samples`, number of rollout targets `n_times`, noise `epsilon`,
    and max step size `dt`.
    Returns: `(sol, time_instants)` where `sol` has shape
    `(n_samples, state_dim, n_times + 1)` and `time_instants` has shape
    `(n_samples, n_times)`.
    """
    qub = system_parameters["qub"]
    qlb = system_parameters["qlb"]
    piub = system_parameters.get("piub", system_parameters["qub"])
    pilb = system_parameters.get("pilb", system_parameters["qlb"])

    def fun(t, y):
        """RHS callback used by SciPy ODE integration.

        Inputs: scalar time `t` and state `y`.
        Returns: numpy derivative vector dy/dt.
        """
        q, p = y[: vec.ndim_total // 2], y[vec.ndim_total // 2 :]
        q = torch.tensor(q, dtype=dtype).unsqueeze(0)
        p = torch.tensor(p, dtype=dtype).unsqueeze(0)
        q_grad, p_grad = vec.eval_vec_field(q, p)
        return np.concatenate([q_grad.squeeze(0).numpy(), p_grad.squeeze(0).numpy()], axis=0)

    # Irregular increments in [0, dt], then cumulative times.
    time_instants = torch.rand(n_samples, n_times, dtype=dtype) * dt
    time_instants = torch.cumsum(time_instants, dim=1)

    q = torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (qub - qlb) + qlb
    p = torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (piub - pilb) + pilb
    if vec.isa_doubled_variables_system:
        q = torch.cat((q, q), dim=1)
        p = torch.cat((p, -p), dim=1)

    initial_conditions = torch.cat((q, p), dim=1)
    sol = np.zeros((n_samples, vec.ndim_total, n_times + 1), dtype=np.float32)
    sol[:, :, 0] = initial_conditions.detach().cpu().numpy()

    for i in range(n_samples):
        y0 = initial_conditions[i].detach().cpu().numpy()
        ti = time_instants[i].detach().cpu().numpy()
        sol_i = solve_ivp(
            fun,
            t_span=[0.0, float(ti[-1])],
            t_eval=ti,
            y0=y0,
            method="RK45",
            atol=1e-10,
            rtol=1e-10,
        ).y
        if epsilon > 0:
            sol_i = sol_i + epsilon * np.random.randn(*sol_i.shape)
        sol[i, :, 1:] = sol_i

    return sol, time_instants


class SRNNDataset(Dataset):
    """Dataset of SRNN initial conditions, times, and target updates."""

    def __init__(self, sol: np.ndarray, times: torch.Tensor):
        """Build tensors from generated trajectories.

        Inputs: `sol` trajectory array and `times` tensor.
        Returns: None.
        """
        self.ics = torch.from_numpy(sol[:, :, 0].astype(np.float32))
        self.updates = torch.from_numpy(sol[:, :, 1:].astype(np.float32))
        self.times = times.to(torch.float32)

    def __len__(self) -> int:
        """Return dataset size.

        Inputs: none.
        Returns: number of samples.
        """
        return len(self.ics)

    def __getitem__(self, idx: int):
        """Fetch one training example.

        Inputs: `idx` sample index.
        Returns: `(initial_state, times, target_updates)` tensors.
        """
        return self.ics[idx], self.times[idx], self.updates[idx]


def train_srnn(
    model: SRNN,
    dataloader: DataLoader,
    epochs: int,
    device: torch.device,
    max_lr: float = 5e-3,
    min_lr: float = 1e-4,
    show_progress: bool = True,
) -> SRNN:
    """Train an SRNN model on trajectory-update supervision.

    Inputs: `model`, `dataloader`, `epochs`, target `device`, and LR schedule
    bounds.
    Returns: trained `SRNN` model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=min_lr)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        div_factor=max_lr / min_lr,
        total_steps=total_steps,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    model.to(device)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter)

    for _ in epoch_iter:
        model.train()
        running_loss = 0.0
        for ics, times, updates in dataloader:
            optimizer.zero_grad()
            ics = ics.to(device)
            times = times.to(device)
            updates = updates.to(device)

            n_times = times.shape[1]
            state = ics.clone()
            loss = 0.0
            for i in range(n_times):
                if i == 0:
                    dt_step = times[:, i : i + 1]
                else:
                    dt_step = times[:, i : i + 1] - times[:, i - 1 : i]
                state = model(state, dt_step)
                loss = loss + criterion(state, updates[:, :, i]) / n_times

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        if show_progress:
            epoch_iter.set_postfix_str(f"{running_loss / max(len(dataloader), 1):.6f}")

    return model


def main() -> None:
    """CLI entry point for SRNN training sweeps.

    Inputs: parsed command-line arguments.
    Returns: None. Saves trained checkpoints to disk.
    """
    parser = argparse.ArgumentParser(description="Train SRNN models on supervised trajectories.")
    parser.add_argument("--ode-name", default="SimpleHO", choices=["SimpleHO", "DampedHO", "HenonHeiles"])
    parser.add_argument("--Ns", default="1500", help="Comma-separated N values.")
    parser.add_argument("--Ms", default="1", help="Comma-separated M values.")
    parser.add_argument("--epsilons", default="0.0", help="Comma-separated epsilon values.")
    parser.add_argument("--dt", default=0.1, type=float, help="Upper bound for random time increments in data generation.")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--hidden-nodes", default=10, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--model-dir", default="modelsSRNN")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n_list = parse_int_list(args.Ns)
    m_list = parse_int_list(args.Ms)
    eps_list = parse_float_list(args.epsilons)

    torch.manual_seed(0)
    np.random.seed(0)

    system_parameters = get_system_parameters(args.ode_name)
    vec = get_vector_field(system_parameters)
    dtype = torch.float32
    run_device = get_device(args.device)
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in n_list:
        for m in m_list:
            for epsilon in eps_list:
                if not args.quiet:
                    print(f"[train] ode={args.ode_name}, N={n}, M={m}, epsilon={epsilon}")

                sol, time_instants = generate_data_supervised_srnn(
                    vec=vec,
                    system_parameters=system_parameters,
                    dtype=dtype,
                    n_samples=n,
                    n_times=m,
                    epsilon=epsilon,
                    dt=args.dt,
                )
                dataset = SRNNDataset(sol, time_instants)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

                model = SRNN(
                    hidden=args.hidden_nodes,
                    d=vec.ndim_total,
                    dtype=dtype,
                    n_layers=args.nlayers,
                    dt=args.dt,
                )
                train_srnn(
                    model=model,
                    dataloader=dataloader,
                    epochs=args.epochs,
                    device=run_device,
                    show_progress=not args.quiet,
                )

                model_path = output_dir / f"N_{n}_M_{m}_epsilon_{format_eps(epsilon)}.pt"
                torch.save(model.state_dict(), model_path)
                if not args.quiet:
                    print(f"[saved] {model_path}")


if __name__ == "__main__":
    main()
