import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.model_quality_core import get_device, get_system_parameters, get_vector_field
from scripts.sampling import sample_ic
from scripts.srnn_model import SRNN
from scripts.utils import SUBSTEPS_PER_DT, solution_scipy

CSV_COLUMNS = [
    "N",
    "M",
    "epsilon",
    "name_experiment",
    "norm dt",
    "norm 10*dt",
    "norm 100*dt",
    "energy dt",
    "energy 10*dt",
    "energy 100*dt",
    "rel norm dt",
    "rel norm 10*dt",
    "rel norm 100*dt",
    "rel energy dt",
    "rel energy 10*dt",
    "rel energy 100*dt",
]


def parse_int_list(values: str) -> list[int]:
    """Parse a comma-separated string into integers.

    Inputs: `values` (str), e.g. "100,500,1000".
    Returns: list[int].
    """
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_list(values: str) -> list[float]:
    """Parse a comma-separated string into floats.

    Inputs: `values` (str), e.g. "0.0,0.01,0.02".
    Returns: list[float].
    """
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def format_eps(epsilon: float) -> str:
    """Format epsilon for checkpoint filenames.

    Inputs: `epsilon` (float).
    Returns: normalized string representation.
    """
    return str(float(epsilon))


def default_output_csv(vary: str) -> str:
    """Return default output CSV path for a given sweep.

    Inputs: `vary` in {"epsilon", "N", "M"}.
    Returns: output CSV path (str).
    """
    if vary == "epsilon":
        return "modelsSRNN/resultsVaryingEpsilon_srnn.csv"
    if vary == "N":
        return "modelsSRNN/resultsVaryingN_srnn.csv"
    if vary == "M":
        return "modelsSRNN/resultsVaryingM_srnn.csv"
    raise ValueError(f"Unsupported vary mode '{vary}'.")


def default_sweep_lists(vary: str) -> tuple[list[float], list[int], list[int]]:
    """Return default epsilon/N/M lists for a sweep.

    Inputs: `vary` in {"epsilon", "N", "M"}.
    Returns: tuple `(epsilons, n_list, m_list)`.
    """
    if vary == "epsilon":
        return [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05], [1500], [1]
    if vary == "N":
        return [0.0], [100, 500, 1000, 1500, 2000], [1]
    if vary == "M":
        return [0.0], [1500], [1, 3, 5, 8]
    raise ValueError(f"Unsupported vary mode '{vary}'.")


def write_results(path: str, rows: list[dict]) -> None:
    """Write evaluation rows to CSV.

    Inputs: destination `path` and list of row dicts.
    Returns: None (writes file to disk).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_srnn_checkpoint(
    model_root: str | Path,
    n: int,
    m: int,
    epsilon: float,
) -> str:
    """Resolve the exact SRNN checkpoint path.

    Inputs: model root, `n`, `m`, and `epsilon`.
    Returns: checkpoint path (str) if exact match exists.
    Raises: `FileNotFoundError` if the exact file is missing.
    """
    root = Path(model_root)
    eps = format_eps(epsilon)
    exact = root / f"N_{n}_M_{m}_epsilon_{eps}.pt"
    if exact.exists():
        return str(exact)

    raise FileNotFoundError(
        f"Could not find SRNN checkpoint for N={n}, M={m}, epsilon={eps} in {root}. "
        f"Expected exact file: {exact.name}"
    )


def eval_hamiltonian(vec, sol: np.ndarray) -> np.ndarray:
    """Evaluate Hamiltonian along a trajectory.

    Inputs: vector field `vec` and trajectory `sol` with shape `(d, T)`.
    Returns: energy values with shape `(T,)`.
    """
    q = sol[0 : vec.ndim_total // 2].T
    pi = sol[vec.ndim_total // 2 :].T
    return vec.eval_hamiltonian(q, pi).reshape(-1)


def approximate_solution_srnn(
    y0: torch.Tensor,
    model: SRNN,
    time: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out SRNN sequentially on a fine substep grid.

    Inputs: initial state `y0`, `model`, coarse `time` array, tensor `dtype`,
    and target `device`.
    Returns: `(sol, time_tensor)` where `sol` has shape `(T, d)`.
    """
    with torch.no_grad():
        t0 = float(time[0])
        tf = float(time[-1])
        d = len(y0)
        dt = float(time[1] - time[0])

        n_steps = int(round((tf - t0) / dt))
        tf = t0 + n_steps * dt

        fine_time = np.linspace(t0, tf, SUBSTEPS_PER_DT * n_steps + 1)
        dt_finer = float(fine_time[1] - fine_time[0])

        time_tensor = torch.from_numpy(fine_time.astype(np.float32)).to(device)
        sol = torch.zeros((len(time_tensor), d), dtype=dtype, device=device)
        sol[0] = y0

        for i in range(len(time_tensor) - 1):
            sol[i + 1] = model(
                sol[i : i + 1],
                torch.tensor([[dt_finer]], dtype=dtype, device=device),
            )[0]

    return sol, time_tensor


def generate_solutions_srnn(
    vec,
    q0: torch.Tensor,
    pi0: torch.Tensor,
    tf: float,
    model: SRNN,
    dtype: torch.dtype,
    device: torch.device,
    model_dt: float,
):
    """Generate SRNN and SciPy trajectories from one initial condition.

    Inputs: system/vector field data, initial condition `(q0, pi0)`, final time
    `tf`, SRNN `model`, tensor `dtype`, compute `device`, and `model_dt`.
    Returns: `(vec, t_eval, sol_scipy, sol_network)`.
    """
    y0 = torch.cat((q0, pi0))

    if vec.isa_doubled_variables_system:
        y0_np = torch.cat((q0[: vec.ndim_total // 4], pi0[: vec.ndim_total // 4])).detach().cpu().numpy()
    else:
        y0_np = y0.detach().cpu().numpy()

    base_t_eval = np.linspace(0.0, tf, int(round(tf / model_dt)) + 1)
    sol_network, t_eval = approximate_solution_srnn(y0.to(device), model, base_t_eval, dtype, device)
    t_eval = t_eval.detach().cpu().numpy()
    sol_network = sol_network.detach().cpu().numpy().T.squeeze()
    sol_scipy = solution_scipy(y0_np, t_eval=t_eval, vec=vec)
    return vec, t_eval, sol_scipy, sol_network


@dataclass
class SRNNQualityResult:
    """Container for averaged SRNN quality metrics.

    Inputs: metric arrays computed over test samples.
    Returns: dataclass instance with mean norm/energy errors.
    """
    avg_norm_differences: np.ndarray
    avg_energy_variation: np.ndarray
    avg_relative_error: np.ndarray
    avg_relative_energy_variation: np.ndarray


def evaluate_srnn_quality(
    ode_name: str,
    model_path: str,
    n_samples: int = 100,
    model_dt: float = 0.1,
    hidden_nodes: int = 10,
    nlayers: int = 5,
    device: str | None = None,
    time_indices: tuple[int, int, int] | None = None,
    show_progress: bool = True,
) -> SRNNQualityResult:
    """Evaluate one SRNN checkpoint on sampled test initial conditions.

    Inputs: model/config arguments including `ode_name`, `model_path`,
    `n_samples`, `model_dt`, architecture, and optional `time_indices`.
    Returns: `SRNNQualityResult` with averaged absolute/relative errors.
    """
    system_parameters = get_system_parameters(ode_name)
    vec = get_vector_field(system_parameters)

    torch.manual_seed(1)
    np.random.seed(1)
    run_device = get_device(device)
    dtype = torch.float32

    model = SRNN(
        hidden=hidden_nodes,
        d=vec.ndim_total,
        dtype=dtype,
        n_layers=nlayers,
        dt=model_dt,
    )
    model.to(run_device)
    model.load_state_dict(torch.load(model_path, map_location=run_device), strict=True)

    z, _ = sample_ic(
        system_parameters=system_parameters,
        vec=vec,
        dtype=dtype,
        n_samples=n_samples,
        dt=model.dt,
        factor=1.0,
        t0=-0.1,
    )
    z = z.to(run_device)

    if time_indices is None:
        time_indices = (
            SUBSTEPS_PER_DT,
            10 * SUBSTEPS_PER_DT,
            100 * SUBSTEPS_PER_DT,
        )
    idx = np.array(list(time_indices), dtype=int)
    tf = np.max(idx) * (model_dt / SUBSTEPS_PER_DT)

    q0 = z[0, : vec.ndim_total // 2]
    pi0 = z[0, vec.ndim_total // 2 :]
    vec, t_eval, sol_scipy, sol_network = generate_solutions_srnn(
        vec,
        q0,
        pi0,
        tf,
        model,
        dtype,
        run_device,
        model_dt=model_dt,
    )

    if np.max(idx) >= sol_network.shape[1]:
        raise ValueError(
            "time_indices exceed the generated trajectory length. "
            f"Max index={np.max(idx)}, trajectory length={sol_network.shape[1]}."
        )

    sols_network = np.zeros((len(z), *sol_network.shape))
    sols_scipy = np.zeros((len(z), *sol_scipy.shape))
    energies_network = np.zeros((len(z), sol_network.shape[1]))
    sols_network[0] = sol_network
    sols_scipy[0] = sol_scipy
    energies_network[0] = eval_hamiltonian(vec, sol_network)

    iterator = range(1, len(z))
    if show_progress:
        iterator = tqdm(iterator)

    for i in iterator:
        q0 = z[i, : vec.ndim_total // 2]
        pi0 = z[i, vec.ndim_total // 2 :]
        vec, t_eval, sol_scipy, sol_network = generate_solutions_srnn(
            vec,
            q0,
            pi0,
            tf,
            model,
            dtype,
            run_device,
            model_dt=model_dt,
        )
        sols_network[i] = sol_network
        sols_scipy[i] = sol_scipy
        energies_network[i] = eval_hamiltonian(vec, sol_network)

    differences_sols = sols_network - sols_scipy
    e0 = energies_network[:, 0:1]
    energy_variations = np.abs(energies_network - e0)

    energy_variations = energy_variations[:, idx]
    differences_sols = differences_sols[:, :, idx]
    norm_differences = np.linalg.norm(differences_sols, axis=1, ord=2)
    denom = np.linalg.norm(sols_scipy[:, :, idx], axis=1, ord=2)
    relative_errors = norm_differences / np.maximum(denom, 1e-12)
    relative_energy_variation = energy_variations / np.maximum(np.abs(e0), 1e-12)

    return SRNNQualityResult(
        avg_norm_differences=np.mean(norm_differences, axis=0),
        avg_energy_variation=np.mean(energy_variations, axis=0),
        avg_relative_error=np.mean(relative_errors, axis=0),
        avg_relative_energy_variation=np.mean(relative_energy_variation, axis=0),
    )


def main() -> None:
    """CLI entry point for SRNN quality sweeps.

    Inputs: parsed command-line arguments.
    Returns: None. Writes a CSV with evaluation metrics.
    """
    parser = argparse.ArgumentParser(description="Evaluate SRNN model quality across a sweep and export CSV.")
    parser.add_argument("--ode-name", default="SimpleHO", choices=["SimpleHO", "DampedHO", "HenonHeiles"])
    parser.add_argument("--vary", default="epsilon", choices=["epsilon", "N", "M"])
    parser.add_argument("--epsilons", default=None, help="Comma-separated epsilon values.")
    parser.add_argument("--Ns", default=None, help="Comma-separated N values.")
    parser.add_argument("--Ms", default=None, help="Comma-separated M values.")
    parser.add_argument("--name-experiment", default="srnn")
    parser.add_argument("--n-samples", default=100, type=int)
    parser.add_argument("--model-root", default="modelsSRNN")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model-dt", default=0.1, type=float)
    parser.add_argument("--hidden-nodes", default=10, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    eps_default, n_default, m_default = default_sweep_lists(args.vary)
    epsilons = parse_float_list(args.epsilons) if args.epsilons else eps_default
    n_list = parse_int_list(args.Ns) if args.Ns else n_default
    m_list = parse_int_list(args.Ms) if args.Ms else m_default
    output_csv = args.output_csv or default_output_csv(args.vary)

    rows: list[dict] = []
    for n in n_list:
        for m in m_list:
            for epsilon in epsilons:
                try:
                    model_path = resolve_srnn_checkpoint(args.model_root, n=n, m=m, epsilon=epsilon)
                except FileNotFoundError as exc:
                    if args.skip_missing:
                        print(f"[skip] {exc}")
                        continue
                    raise

                print(f"N={n}, M={m}, epsilon={epsilon}, model={model_path}")
                result = evaluate_srnn_quality(
                    ode_name=args.ode_name,
                    model_path=model_path,
                    n_samples=args.n_samples,
                    model_dt=args.model_dt,
                    hidden_nodes=args.hidden_nodes,
                    nlayers=args.nlayers,
                    device=args.device,
                    show_progress=not args.quiet,
                )

                rows.append(
                    {
                        "N": n,
                        "M": m,
                        "epsilon": epsilon,
                        "name_experiment": args.name_experiment,
                        "norm dt": result.avg_norm_differences[0],
                        "norm 10*dt": result.avg_norm_differences[1],
                        "norm 100*dt": result.avg_norm_differences[2],
                        "energy dt": result.avg_energy_variation[0],
                        "energy 10*dt": result.avg_energy_variation[1],
                        "energy 100*dt": result.avg_energy_variation[2],
                        "rel norm dt": result.avg_relative_error[0],
                        "rel norm 10*dt": result.avg_relative_error[1],
                        "rel norm 100*dt": result.avg_relative_error[2],
                        "rel energy dt": result.avg_relative_energy_variation[0],
                        "rel energy 10*dt": result.avg_relative_energy_variation[1],
                        "rel energy 100*dt": result.avg_relative_energy_variation[2],
                    }
                )

    write_results(output_csv, rows)
    print(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
