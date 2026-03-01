import argparse
import csv
from pathlib import Path

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

    Inputs: `values` string like "10,50,100".
    Returns: list[int].
    """
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_list(values: str) -> list[float]:
    """Parse a comma-separated string into floats.

    Inputs: `values` string like "0.0,0.01,0.02".
    Returns: list[float].
    """
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def parse_str_list(values: str) -> list[str]:
    """Parse a comma-separated string into tokens.

    Inputs: `values` string.
    Returns: list[str] of stripped items.
    """
    return [v.strip() for v in values.split(",") if v.strip()]


def parse_bool(value: str) -> bool:
    """Parse common textual boolean representations.

    Inputs: `value` string (e.g. "true", "0", "yes").
    Returns: boolean value.
    Raises: `ValueError` for unsupported strings.
    """
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value '{value}'.")


def default_output_csv(vary: str, is_supervised: bool) -> str:
    """Return default output CSV path for a sweep/mode.

    Inputs: sweep key `vary` and mode flag `is_supervised`.
    Returns: default output CSV path.
    """
    base_dir = "supervisedNetworks" if is_supervised else "unsupervisedNetworks"
    if vary == "epsilon":
        return f"{base_dir}/resultsVaryingEpsilon.csv"
    if vary == "N":
        return f"{base_dir}/resultsVaryingN.csv"
    if vary == "M":
        return f"{base_dir}/resultsVaryingM.csv"
    raise ValueError(f"Unsupported vary mode '{vary}'.")


def default_model_root(is_supervised: bool) -> str:
    """Return default checkpoint root for supervised/unsupervised mode.

    Inputs: `is_supervised` flag.
    Returns: model root directory path.
    """
    if is_supervised:
        return "supervisedNetworks/savedModels"
    return "unsupervisedNetworks/savedModels"


def default_sweep_lists(vary: str) -> tuple[list[float], list[int], list[int]]:
    """Return default epsilon/N/M sweep lists for selected axis.

    Inputs: sweep key `vary`.
    Returns: tuple `(epsilons, n_list, m_list)`.
    """
    if vary == "epsilon":
        return [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05], [100], [50]
    if vary == "N":
        return [0.0], [10, 50, 100, 150, 200], [50]
    if vary == "M":
        return [0.0], [100], [10, 30, 50, 80]
    raise ValueError(f"Unsupported vary mode '{vary}'.")


def write_results(path: str, rows: list[dict]) -> None:
    """Write evaluation rows to CSV.

    Inputs: destination `path` and row dictionaries.
    Returns: None. Writes file to disk.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point for model-quality evaluation sweeps.

    Inputs: command-line arguments.
    Returns: None. Writes a CSV with aggregated metrics.
    """
    parser = argparse.ArgumentParser(description="Evaluate model quality across a sweep and export CSV.")
    parser.add_argument("--ode-name", default="SimpleHO", choices=["SimpleHO", "DampedHO", "HenonHeiles"])
    parser.add_argument("--vary", default="epsilon", choices=["epsilon", "N", "M"])
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment names. If omitted, inferred from --is_supervised.")
    parser.add_argument(
        "--is_supervised",
        action="store_true",
        help=(
            "Use supervised setting: experiments=['pinn','sympflow'] and nlayers=5. "
            "Default is unsupervised setting with experiments=['pinnReg','pinnNoReg','hamReg','noHamReg'] and nlayers=3."
        ),
    )
    parser.add_argument("--cases-file", default=None, help="Optional CSV file listing explicit model files to evaluate. If set, filename resolution from epsilon/N/M is skipped.")
    parser.add_argument("--epsilons", default=None, help="Comma-separated epsilon values.")
    parser.add_argument("--Ns", default=None, help="Comma-separated N values.")
    parser.add_argument("--Ms", default=None, help="Comma-separated M values.")
    parser.add_argument("--n-samples", default=100, type=int)
    parser.add_argument("--model-root", default=None, help="Checkpoint root directory. If omitted, inferred from --is_supervised.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model-dt", default=1.0, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    from scripts.model_quality_core import evaluate_model_quality, resolve_model_checkpoint

    if args.experiments:
        experiments = parse_str_list(args.experiments)
    elif args.is_supervised:
        experiments = ["pinn", "sympflow"]
    else:
        experiments = ["pinnReg", "pinnNoReg", "hamReg", "noHamReg"]

    nlayers = 5 if args.is_supervised else 3
    eps_default, n_default, m_default = default_sweep_lists(args.vary)

    epsilons = parse_float_list(args.epsilons) if args.epsilons else eps_default
    n_list = parse_int_list(args.Ns) if args.Ns else n_default
    m_list = parse_int_list(args.Ms) if args.Ms else m_default

    if args.output_csv is None:
        output_csv = default_output_csv(args.vary, is_supervised=args.is_supervised)
    else:
        output_csv = args.output_csv

    if args.model_root is None:
        model_root = default_model_root(is_supervised=args.is_supervised)
    else:
        model_root = args.model_root

    rows: list[dict] = []

    if args.cases_file:
        cases_path = Path(args.cases_file)
        with cases_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for idx, case in enumerate(reader, start=1):
                model_path = (case.get("model_path") or "").strip()
                if not model_path:
                    raise ValueError(
                        f"Missing required 'model_path' at row {idx} in {cases_path}."
                    )

                ode_name = (case.get("ode_name") or args.ode_name).strip()
                experiment = (case.get("name_experiment") or experiments[0]).strip()

                n_value = int((case.get("N") or n_list[0]))
                m_value = int((case.get("M") or m_list[0]))
                epsilon_value = float((case.get("epsilon") or epsilons[0]))

                n_samples_value = int((case.get("n_samples") or args.n_samples))
                model_dt_value = float((case.get("model_dt") or args.model_dt))
                nlayers_value = int((case.get("nlayers") or nlayers))
                raw_is_regular_grid = (case.get("is_regular_grid") or "").strip()
                is_regular_grid_value = parse_bool(raw_is_regular_grid) if raw_is_regular_grid else False

                has_layer_case = "nlayers" in case and (case.get("nlayers") or "").strip()
                if has_layer_case:
                    print(
                        f"[case {idx}] n_layers={nlayers_value}, experiment={experiment}, "
                        f"model={model_path}, is_supervised={args.is_supervised}"
                    )
                else:
                    print(
                        f"[case {idx}] N={n_value}, M={m_value}, epsilon={epsilon_value}, experiment={experiment}, "
                        f"model={model_path}, nlayers={nlayers_value}, is_supervised={args.is_supervised}"
                    )

                try:
                    result = evaluate_model_quality(
                        ode_name=ode_name,
                        name_experiment=experiment,
                        model_path=model_path,
                        n_samples=n_samples_value,
                        is_regular_grid=is_regular_grid_value,
                        model_dt=model_dt_value,
                        nlayers=nlayers_value,
                        device=args.device,
                        show_progress=not args.quiet,
                    )
                except FileNotFoundError as exc:
                    if args.skip_missing:
                        print(f"[skip] {exc}")
                        continue
                    raise

                rows.append(
                    {
                        "N": n_value,
                        "M": m_value,
                        "epsilon": epsilon_value,
                        "name_experiment": experiment,
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
    else:
        for n in n_list:
            for m in m_list:
                for epsilon in epsilons:
                    for experiment in experiments:
                        try:
                            model_path = resolve_model_checkpoint(
                                model_root=model_root,
                                ode_name=args.ode_name,
                                name_experiment=experiment,
                                n=n,
                                m=m,
                                epsilon=epsilon,
                            )
                        except FileNotFoundError as exc:
                            if args.skip_missing:
                                print(f"[skip] {exc}")
                                continue
                            raise

                        print(
                            f"N={n}, M={m}, epsilon={epsilon}, experiment={experiment}, "
                            f"model={model_path}, nlayers={nlayers}, is_supervised={args.is_supervised}"
                        )

                        result = evaluate_model_quality(
                            ode_name=args.ode_name,
                            name_experiment=experiment,
                            model_path=model_path,
                            n_samples=args.n_samples,
                            is_regular_grid=False,
                            model_dt=args.model_dt,
                            nlayers=nlayers,
                            device=args.device,
                            show_progress=not args.quiet,
                        )

                        rows.append(
                            {
                                "N": n,
                                "M": m,
                                "epsilon": epsilon,
                                "name_experiment": experiment,
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
