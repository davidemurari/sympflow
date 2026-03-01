import argparse
import glob
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm


def get_poincare_section(orbit):
    """Convert one orbit to Poincare points `(q_y, p_y)` at x=0 crossings.

    Inputs: `orbit` array with shape `(4, T)`.
    Returns: tuple of Python lists `(q_y_points, p_y_points)`.
    """
    x, y, _, y_dot = orbit[0], orbit[1], orbit[2], orbit[3]
    y_poincare = []
    y_dot_poincare = []
    for i in range(len(x) - 1):
        if x[i] * x[i + 1] < 0.0:
            y_poincare.append(0.5 * (y[i] + y[i + 1]))
            y_dot_poincare.append(0.5 * (y_dot[i] + y_dot[i + 1]))
    return y_poincare, y_dot_poincare


def get_random_intial_conditons(energy):
    """Sample one random Hénon-Heiles initial condition at fixed energy.

    Inputs: target `energy` level.
    Returns: numpy array `[q1, q2, p1, p2]`.
    """
    result = False
    while not result:
        with np.errstate(invalid="raise"):
            try:
                q1, q2 = 0.3 * (1 - 2 * np.random.random(2))
                p1 = 0.2 * (1 - 2 * np.random.random())
                p2 = abs(np.sqrt(2 * (energy - (q1**2 * q2 - q2**3 / 3.0)) - (q1**2 + q2**2 + p1**2)))
                result = True
            except FloatingPointError:
                continue
    return np.array([q1, q2, p1, p2])


def solution_scipy(y0, t_eval, vec):
    """Compute one reference trajectory with SciPy RK45.

    Inputs: initial state `y0`, time grid `t_eval`, and vector field `vec`.
    Returns: `(solution, times)` where solution has shape `(d, T)`.
    """

    from scipy.integrate import solve_ivp
    import torch

    def fun(_t, y):
        q, p = y[: vec.ndim_total // 2], y[vec.ndim_total // 2 :]
        q = torch.tensor(q).unsqueeze(0)
        p = torch.tensor(p).unsqueeze(0)
        q_grad, p_grad = vec.eval_vec_field(q, p)
        return np.concatenate([q_grad.squeeze(0).numpy(), p_grad.squeeze(0).numpy()], axis=0)

    t_span = [t_eval[0], t_eval[-1]]
    res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, method="RK45", rtol=1e-12)
    return res.y, res.t


def get_last_trained_model(path):
    """Return latest timestamped `trained_model_*.pt` under one directory.

    Inputs: directory `path`.
    Returns: latest checkpoint path.
    """
    files = glob.glob(os.path.join(path, "trained_model*.pt"))
    if not files:
        raise FileNotFoundError(f"No trained_model_*.pt found in {path}")

    def extract_timestamp(filename):
        basename = os.path.basename(filename)
        match = re.search(r"(\d{8}_\d{6})(?=\.pt$)", basename)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return datetime.fromtimestamp(os.path.getmtime(filename))

    return max(files, key=extract_timestamp)


def approximate_solution(y0, model, t0, tf, fine_resolution, dtype, device):
    """Roll out one model on `[t0, tf]` with per-interval fine substeps.

    Inputs: initial state, model, time bounds, fine resolution, and tensor config.
    Returns: `(sol, time)` where `sol` has shape `(T, d)`.
    """
    import torch

    with torch.no_grad():
        d = len(y0)
        dt = model.dt

        tf = (tf // dt) * dt
        num_intervals = int((tf - t0) / dt)
        ref_interval = torch.linspace(0, dt, fine_resolution, device=device)[1:]
        jump = len(ref_interval)

        sol = torch.zeros((num_intervals * jump + 1, d), dtype=dtype, device=device)
        sol[0] = y0
        time = torch.linspace(t0, tf, jump * num_intervals + 1, device=device)

        intermediate_sols = torch.zeros((num_intervals, d), device=device)
        intermediate_sols[0] = y0

        for i in range(1, num_intervals):
            intermediate_sols[i] = model(
                intermediate_sols[i - 1 : i],
                torch.tensor([[dt]], dtype=dtype, device=device),
            )[0]

        times_global = torch.kron(torch.ones(num_intervals, device=device), ref_interval)
        initial_conditions = intermediate_sols.repeat_interleave(jump, dim=0)
        predictions = model(initial_conditions, times_global)
        sol[1:] = predictions
    return sol, time


def parse_str_list(values: str | None) -> list[str]:
    """Parse a comma-separated string into tokens.

    Inputs: optional `values` string.
    Returns: list of non-empty stripped strings.
    """
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def default_experiments(is_supervised: bool) -> list[str]:
    """Return default experiment names for supervised/unsupervised mode.

    Inputs: `is_supervised` mode flag.
    Returns: ordered experiment list.
    """
    if is_supervised:
        return ["pinn", "sympflow"]
    return ["pinnReg", "pinnNoReg", "hamReg", "noHamReg", "mixed"]


def default_model_root(is_supervised: bool) -> str:
    """Return default model root for supervised/unsupervised mode.

    Inputs: `is_supervised` mode flag.
    Returns: model root path.
    """
    if is_supervised:
        return "supervisedNetworks/savedModels"
    return "unsupervisedNetworks/savedModels"


def default_output_dir(is_supervised: bool) -> str:
    """Return default Poincare output directory.

    Inputs: `is_supervised` mode flag.
    Returns: output directory path.
    """
    if is_supervised:
        return "supervisedNetworks/figures/poincareSections"
    return "unsupervisedNetworks/figures/poincareSections"


def build_model(name_experiment: str, model_parameters: dict, vec, model_dt: float):
    """Instantiate the right network class for one experiment name.

    Inputs: experiment name, model config, vector field, and model dt.
    Returns: instantiated torch model.
    """
    from scripts.networks import genericNet, sympNet

    if name_experiment in {"sympflow", "hamReg", "noHamReg", "mixed"}:
        return sympNet(model_parameters, vec=vec, dt=model_dt)
    if name_experiment in {"pinn", "pinnReg", "pinnNoReg"}:
        return genericNet(model_parameters, vec=vec, dt=model_dt)
    raise ValueError(f"Unsupported experiment '{name_experiment}'.")


def resolve_checkpoint(model_root: str, name_experiment: str) -> str:
    """Resolve the latest checkpoint for one Hénon-Heiles experiment.

    Inputs: `model_root` and `name_experiment`.
    Returns: checkpoint path.
    """
    path = Path(model_root) / "HenonHeiles" / name_experiment
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return get_last_trained_model(str(path))


def make_initial_conditions(n_orbits: int, energy: float, seed: int) -> list[np.ndarray]:
    """Generate random initial conditions for one energy shell.

    Inputs: number of orbits, target energy, and RNG seed.
    Returns: list of state vectors `[q1, q2, p1, p2]`.
    """
    np.random.seed(seed)
    return [get_random_intial_conditons(energy) for _ in range(n_orbits)]


def compute_reference_orbits(vec, ics: list[np.ndarray], t_eval: np.ndarray, show_progress: bool) -> list[np.ndarray]:
    """Integrate reference trajectories with SciPy.

    Inputs: vector field, initial conditions, evaluation times, progress flag.
    Returns: list of trajectories with shape `(d, T)`.
    """
    iterator = ics
    if show_progress:
        iterator = tqdm(ics, desc="reference", leave=False)

    orbit_set: list[np.ndarray] = []
    for y0 in iterator:
        sol, _ = solution_scipy(y0, t_eval, vec)
        orbit_set.append(sol)
    return orbit_set


def compute_model_orbits(
    model,
    ics: list[np.ndarray],
    t0: float,
    tf: float,
    fine_resolution: int,
    dtype,
    device,
    show_progress: bool,
) -> list[np.ndarray]:
    """Roll out one neural model over all initial conditions.

    Inputs: model, IC list, time bounds, substep count, tensor settings, progress flag.
    Returns: list of trajectories with shape `(d, T)`.
    """
    import torch

    iterator = ics
    if show_progress:
        iterator = tqdm(ics, desc="model", leave=False)

    orbit_set: list[np.ndarray] = []
    for y0 in iterator:
        sol, _ = approximate_solution(
            y0=torch.from_numpy(y0.astype(np.float32)),
            model=model,
            t0=t0,
            tf=tf,
            fine_resolution=fine_resolution,
            dtype=dtype,
            device=device,
        )
        orbit_set.append(sol.T.detach().cpu().numpy())
    return orbit_set


def to_poincare_points(orbit_set: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert trajectories to Poincare points `(q_y, p_y)`.

    Inputs: list of trajectories.
    Returns: list of tuples `(qy_points, py_points)`.
    """
    points: list[tuple[np.ndarray, np.ndarray]] = []
    for orbit in orbit_set:
        qy, py = get_poincare_section(orbit)
        points.append((np.asarray(qy), np.asarray(py)))
    return points


def infer_limits(all_points: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float, float]:
    """Infer axis limits from all Poincare points.

    Inputs: list of `(qy, py)` arrays.
    Returns: `(xmin, xmax, ymin, ymax)`.
    """
    xs = [x for x, _ in all_points if x.size > 0]
    ys = [y for _, y in all_points if y.size > 0]
    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)
    xmin = min(float(np.min(x)) for x in xs)
    xmax = max(float(np.max(x)) for x in xs)
    ymin = min(float(np.min(y)) for y in ys)
    ymax = max(float(np.max(y)) for y in ys)
    pad_x = 0.05 * max(1e-12, xmax - xmin)
    pad_y = 0.05 * max(1e-12, ymax - ymin)
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def expand_limits(
    limits: tuple[float, float, float, float],
    pad_frac: float = 0.25,
    min_abs_pad: float = 0.1,
) -> tuple[float, float, float, float]:
    """Expand plot limits by a relative and absolute padding.

    Inputs: base `limits=(xmin,xmax,ymin,ymax)`, relative `pad_frac`,
    and absolute minimum pad.
    Returns: expanded limits.
    """
    xmin, xmax, ymin, ymax = limits
    dx = max(1e-12, xmax - xmin)
    dy = max(1e-12, ymax - ymin)
    pad_x = max(min_abs_pad, pad_frac * dx)
    pad_y = max(min_abs_pad, pad_frac * dy)
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def plot_single(points, title: str, out_path: Path, limits: tuple[float, float, float, float], use_tex: bool, dpi: int) -> None:
    """Plot one Poincare panel and save it.

    Inputs: points list, title, output path, axis limits, TeX toggle, and dpi.
    Returns: None (writes PDF).
    """
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["text.usetex"] = use_tex
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10

    colors = plt.cm.plasma(np.linspace(0, 1, max(1, len(points))))
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.2))
    ax.set_title(title)
    for i, (x, y) in enumerate(points):
        if x.size == 0:
            continue
        ax.plot(x, y, ".", color=colors[i], markersize=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$q_y$")
    ax.set_ylabel(r"$p_y$")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    panels: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]],
    out_path: Path,
    limits: tuple[float, float, float, float],
    use_tex: bool,
    dpi: int,
) -> None:
    """Plot a horizontal panel with multiple Poincare sections.

    Inputs: panel `(title, points)` list, output path, shared limits, TeX toggle, and dpi.
    Returns: None (writes PDF).
    """
    if not panels:
        return

    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["text.usetex"] = use_tex
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10

    ncols = len(panels)
    fig, axs = plt.subplots(1, ncols, figsize=(2.6 * ncols, 2.4), sharex=True, sharey=True)
    if ncols == 1:
        axs = [axs]

    for ax, (title, points) in zip(axs, panels):
        colors = plt.cm.plasma(np.linspace(0, 1, max(1, len(points))))
        ax.set_title(title)
        for i, (x, y) in enumerate(points):
            if x.size == 0:
                continue
            ax.plot(x, y, ".", color=colors[i], markersize=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel(r"$q_y$")

    axs[0].set_ylabel(r"$p_y$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point for Hénon-Heiles Poincare section generation.

    Inputs: command-line arguments.
    Returns: None. Saves PDF plots to disk.
    """
    parser = argparse.ArgumentParser(
        description="Generate Hénon-Heiles Poincare sections for reference and trained models."
    )
    parser.add_argument("--is-supervised", action="store_true", help="Use supervised model defaults (pinn,sympflow).")
    parser.add_argument("--experiments", default=None, help="Comma-separated experiments to evaluate.")
    parser.add_argument("--model-root", default=None, help="Checkpoint root directory.")
    parser.add_argument("--n-orbits", type=int, default=10)
    parser.add_argument("--energy", type=float, default=0.1)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--tf", type=float, default=4000.0)
    parser.add_argument("--time-points", type=int, default=400000)
    parser.add_argument("--fine-resolution", type=int, default=101, help="Substeps per model interval for rollout.")
    parser.add_argument("--model-dt", type=float, default=1.0)
    parser.add_argument("--hidden-nodes", type=int, default=10)
    parser.add_argument("--nlayers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13353)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default="PSection")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Generate only the reference section.")
    parser.add_argument(
        "--limits-pad-frac",
        type=float,
        default=0.25,
        help="Relative padding applied to inferred axis limits.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing checkpoints instead of failing.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    import torch
    from scripts.experiments import Henon_Heiles_exp
    from scripts.vector_fields import HenonHeiles

    if args.reference_only:
        experiments = []
    else:
        experiments = parse_str_list(args.experiments) or default_experiments(args.is_supervised)
    model_root = args.model_root or default_model_root(args.is_supervised)
    output_dir = Path(args.output_dir or default_output_dir(args.is_supervised))

    run_device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    nlayers = args.nlayers if args.nlayers is not None else (5 if args.is_supervised else 3)

    system_parameters = Henon_Heiles_exp
    vec = HenonHeiles(system_parameters)

    t_eval = np.linspace(args.t0, args.tf, args.time_points + 1)
    ics = make_initial_conditions(args.n_orbits, args.energy, args.seed)

    panels: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    all_points: list[tuple[np.ndarray, np.ndarray]] = []
    saved_paths: list[Path] = []
    reference_points: list[tuple[np.ndarray, np.ndarray]] | None = None

    if not args.skip_reference:
        if not args.quiet:
            print("Generating reference Poincare section...")
        ref_orbits = compute_reference_orbits(vec, ics, t_eval, show_progress=not args.quiet)
        ref_points = to_poincare_points(ref_orbits)
        panels.append(("Reference", ref_points))
        all_points.extend(ref_points)
        reference_points = ref_points

    model_parameters = {
        "hidden_nodes": args.hidden_nodes,
        "act_name": "tanh",
        "nlayers": nlayers,
        "device": run_device,
        "dtype": dtype,
        "d": vec.ndim_total,
    }

    for exp in experiments:
        try:
            checkpoint = resolve_checkpoint(model_root=model_root, name_experiment=exp)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[skip] {exc}")
                continue
            raise

        if not args.quiet:
            print(f"Generating Poincare section for {exp} from {checkpoint}")

        model = build_model(exp, model_parameters=model_parameters, vec=vec, model_dt=args.model_dt)
        model.to(run_device)
        model.load_state_dict(torch.load(checkpoint, map_location=run_device), strict=True)

        model_orbits = compute_model_orbits(
            model=model,
            ics=ics,
            t0=args.t0,
            tf=args.tf,
            fine_resolution=args.fine_resolution,
            dtype=dtype,
            device=run_device,
            show_progress=not args.quiet,
        )
        model_points = to_poincare_points(model_orbits)

        panels.append((exp, model_points))
        all_points.extend(model_points)

    if not panels:
        raise ValueError("No panels generated. Check experiments and checkpoints.")

    # Use reference-derived limits for consistency across models, then pad slightly.
    if reference_points is not None:
        limits = expand_limits(infer_limits(reference_points), pad_frac=args.limits_pad_frac)
    else:
        limits = expand_limits(infer_limits(all_points), pad_frac=args.limits_pad_frac)

    for title, points in panels:
        filename = f"{args.output_prefix}_{title}.pdf".replace(" ", "_")
        out_path = output_dir / filename
        plot_single(points, title=title, out_path=out_path, limits=limits, use_tex=args.use_tex, dpi=args.dpi)
        saved_paths.append(out_path)

    mode_tag = "supervised" if args.is_supervised else "unsupervised"
    combined_path = output_dir / f"{args.output_prefix}_combined_{mode_tag}.pdf"
    plot_combined(panels, out_path=combined_path, limits=limits, use_tex=args.use_tex, dpi=args.dpi)
    saved_paths.append(combined_path)

    print("Saved Poincare plots:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
