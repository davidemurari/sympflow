import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from scripts.experiments import DampedHO_exp, Henon_Heiles_exp, SimpleHO_exp
from scripts.networks import genericNet, sympNet
from scripts.sampling import sample_ic
from scripts.utils import SUBSTEPS_PER_DT, generate_solutions
from scripts.vector_fields import DampedHarmonicOscillator, HarmonicOscillator, HenonHeiles


_SYSTEM_PARAMETERS = {
    "SimpleHO": SimpleHO_exp,
    "DampedHO": DampedHO_exp,
    "HenonHeiles": Henon_Heiles_exp,
}

_VECTOR_FIELDS = {
    "HarmonicOscillator": HarmonicOscillator,
    "DampedHarmonicOscillator": DampedHarmonicOscillator,
    "HenonHeiles": HenonHeiles,
}


def normalize_experiment_name(name_experiment: str) -> str:
    if name_experiment in {"sympflowNoReg", "noHamReg", "hamReg", "mixed"}:
        return "sympflow"
    if name_experiment in {"pinnReg", "pinnNoReg"}:
        return "pinn"
    return name_experiment


def get_system_parameters(ode_name: str) -> dict:
    if ode_name not in _SYSTEM_PARAMETERS:
        raise ValueError(f"Unsupported ode_name '{ode_name}'. Expected one of {sorted(_SYSTEM_PARAMETERS.keys())}.")
    return _SYSTEM_PARAMETERS[ode_name]


def get_vector_field(system_parameters: dict):
    vec_name = system_parameters["vec_field_name"]
    if vec_name not in _VECTOR_FIELDS:
        raise ValueError(f"Unsupported vector field '{vec_name}'.")
    return _VECTOR_FIELDS[vec_name](system_parameters)


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_eps(epsilon: float) -> str:
    return str(float(epsilon))


def resolve_model_checkpoint(
    model_root: str | os.PathLike[str],
    ode_name: str,
    name_experiment: str,
    n: int,
    m: int,
    epsilon: float,
) -> str:
    base_raw = Path(model_root) / ode_name / name_experiment
    base_normalized = Path(model_root) / ode_name / normalize_experiment_name(name_experiment)
    eps_str = format_eps(epsilon)
    stem = f"trained_model_eps_{eps_str}_N_{n}_M_{m}"

    candidates: list[str] = []
    searched_dirs = []

    for base in [base_raw, base_normalized]:
        if str(base) in searched_dirs:
            continue
        searched_dirs.append(str(base))

        exact = base / f"{stem}.pt"
        if exact.exists():
            return str(exact)

        pattern = str(base / f"{stem}_*.pt")
        candidates.extend(glob.glob(pattern))

    matches = candidates
    if not matches:
        raise FileNotFoundError(
            f"Could not find checkpoint for experiment='{name_experiment}', N={n}, M={m}, epsilon={eps_str} "
            f"in any of {searched_dirs}. Tried exact '{stem}.pt' and timestamped '{stem}_*.pt'."
        )
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def eval_hamiltonian(vec, sol: np.ndarray) -> np.ndarray:
    q = sol[0 : vec.ndim_total // 2].T
    pi = sol[vec.ndim_total // 2 :].T
    return vec.eval_hamiltonian(q, pi).reshape(-1)


def _build_model(name_experiment: str, model_parameters: dict, vec, dt: float):
    name_experiment = normalize_experiment_name(name_experiment)
    if name_experiment == "pinn":
        return genericNet(model_parameters, vec=vec, dt=dt)
    if name_experiment == "sympflow":
        return sympNet(model_parameters, vec=vec, dt=dt)
    raise ValueError(f"Unsupported experiment '{name_experiment}'. Expected 'pinn' or 'sympflow'.")


@dataclass
class ModelQualityResult:
    sols_network: np.ndarray
    sols_scipy: np.ndarray
    energies_network: np.ndarray
    t_eval: np.ndarray
    time_indices: np.ndarray
    avg_norm_differences: np.ndarray
    avg_energy_variation: np.ndarray
    avg_relative_error: np.ndarray
    avg_relative_energy_variation: np.ndarray


def evaluate_model_quality(
    ode_name: str,
    name_experiment: str,
    model_path: str | os.PathLike[str],
    n_samples: int = 500,
    is_regular_grid: bool = False,
    model_dt: float = 1.0,
    nlayers: int = 5,
    hidden_nodes: int = 10,
    activation: str = "tanh",
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
    time_indices: Iterable[int] | None = None,
    t0_sampling: float = -0.1,
    factor_sampling: float = 1.0,
    show_progress: bool = True,
) -> ModelQualityResult:
    system_parameters = get_system_parameters(ode_name)
    vec = get_vector_field(system_parameters)

    torch.manual_seed(1)
    np.random.seed(1)
    run_device = get_device(device)

    model_parameters = dict(
        hidden_nodes=hidden_nodes,
        act_name=activation,
        nlayers=nlayers,
        device=run_device,
        dtype=dtype,
        d=vec.ndim_total,
    )

    model = _build_model(name_experiment, model_parameters, vec, dt=model_dt)
    model.to(run_device)
    model.load_state_dict(torch.load(model_path, map_location=run_device), strict=True)

    if is_regular_grid:
        if vec.ndim_spatial != 1:
            raise ValueError("Regular-grid mode is only supported for 1D spatial systems.")
        side = int(np.sqrt(n_samples))
        if side * side != n_samples:
            raise ValueError(f"n_samples must be a perfect square for a regular grid, got {n_samples}.")
        x = np.linspace(system_parameters["qlb"], system_parameters["qub"], side)
        y = np.linspace(system_parameters["pilb"], system_parameters["piub"], side)
        xx, yy = np.meshgrid(x, y)
        z = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
        z = torch.tensor(z, device=run_device, dtype=dtype)
    else:
        z, _ = sample_ic(
            system_parameters=system_parameters,
            vec=vec,
            dtype=dtype,
            n_samples=n_samples,
            dt=model.dt,
            factor=factor_sampling,
            t0=t0_sampling,
        )
        z = z.to(run_device)

    if time_indices is None:
        time_indices = (
            SUBSTEPS_PER_DT,
            10 * SUBSTEPS_PER_DT,
            100 * SUBSTEPS_PER_DT,
        )
    time_indices = np.array(list(time_indices), dtype=int)
    tf = np.max(time_indices) * (model_dt / SUBSTEPS_PER_DT)

    q0 = z[0, : vec.ndim_total // 2]
    pi0 = z[0, vec.ndim_total // 2 :]
    vec, t_eval, sol_scipy, sol_network = generate_solutions(vec, q0, pi0, tf, model, dtype, run_device)

    if np.max(time_indices) >= sol_network.shape[1]:
        raise ValueError(
            "time_indices exceed the generated trajectory length. "
            f"Max index={np.max(time_indices)}, trajectory length={sol_network.shape[1]}."
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
        vec, t_eval, sol_scipy, sol_network = generate_solutions(vec, q0, pi0, tf, model, dtype, run_device)
        sols_network[i] = sol_network
        sols_scipy[i] = sol_scipy
        energies_network[i] = eval_hamiltonian(vec, sol_network)

    differences_sols = sols_network - sols_scipy
    e0 = energies_network[:, 0:1]
    energy_variations = np.abs(energies_network - e0)

    energy_variations = energy_variations[:, time_indices]
    differences_sols = differences_sols[:, :, time_indices]
    norm_differences = np.linalg.norm(differences_sols, axis=1, ord=2)
    denom = np.linalg.norm(sols_scipy[:, :, time_indices], axis=1, ord=2)
    relative_errors = norm_differences / np.maximum(denom, 1e-12)

    relative_energy_variation = energy_variations / np.maximum(np.abs(e0), 1e-12)

    avg_norm_differences = np.mean(norm_differences, axis=0)
    avg_energy_variation = np.mean(energy_variations, axis=0)
    avg_relative_error = np.mean(relative_errors, axis=0)
    avg_relative_energy_variation = np.mean(relative_energy_variation, axis=0)

    return ModelQualityResult(
        sols_network=sols_network,
        sols_scipy=sols_scipy,
        energies_network=energies_network,
        t_eval=t_eval,
        time_indices=time_indices,
        avg_norm_differences=avg_norm_differences,
        avg_energy_variation=avg_energy_variation,
        avg_relative_error=avg_relative_error,
        avg_relative_energy_variation=avg_relative_energy_variation,
    )
