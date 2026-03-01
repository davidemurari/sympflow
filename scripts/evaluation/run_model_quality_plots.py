import argparse
import csv
from pathlib import Path


_DEFAULT_SUPERVISED_EXPERIMENTS = ["pinn", "sympflow"]
_DEFAULT_UNSUPERVISED_EXPERIMENTS = ["pinnReg", "pinnNoReg", "hamReg", "noHamReg"]

_EXPERIMENT_LABELS = {
    "pinn": ("Error (MLP)", "Energy (MLP)"),
    "sympflow": ("Error (SympFlow)", "Energy (SympFlow)"),
    "srnn": ("Error (SRNN)", "Energy (SRNN)"),
    "pinnReg": ("Error (MLP, reg)", "Energy (MLP, reg)"),
    "pinnNoReg": ("Error (MLP, no-reg)", "Energy (MLP, no-reg)"),
    "hamReg": ("Error (SympFlow, reg)", "Energy (SympFlow, reg)"),
    "noHamReg": ("Error (SympFlow, no-reg)", "Energy (SympFlow, no-reg)"),
}


def _float(v: str) -> float:
    """Convert a CSV numeric string to float.

    Inputs: `v` string value.
    Returns: parsed float.
    """
    return float(v)


def _int(v: str) -> int:
    """Convert a CSV numeric string to int.

    Inputs: `v` string value (possibly like "10.0").
    Returns: parsed integer.
    """
    return int(float(v))


def parse_str_list(values: str) -> list[str]:
    """Parse a comma-separated string into tokens.

    Inputs: `values` string (e.g. "pinn,sympflow").
    Returns: list of stripped strings.
    """
    return [v.strip() for v in values.split(",") if v.strip()]


def normalize_experiment_name(name_experiment: str) -> str:
    """Normalize legacy experiment names used in old CSV files.

    Inputs: raw experiment name.
    Returns: canonical experiment name.
    """
    # Keep backward compatibility with older CSVs.
    if name_experiment == "sympflowNoReg":
        return "sympflow"
    return name_experiment


def load_rows(path: str) -> list[dict]:
    """Load and type-cast model-quality CSV rows.

    Inputs: CSV file `path`.
    Returns: list of parsed row dictionaries.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    parsed: list[dict] = []
    for row in rows:
        parsed.append(
            {
                "N": _int(row["N"]),
                "M": _int(row["M"]),
                "epsilon": _float(row["epsilon"]),
                "name_experiment": normalize_experiment_name(row["name_experiment"]),
                "norm dt": _float(row["norm dt"]),
                "norm 10*dt": _float(row["norm 10*dt"]),
                "norm 100*dt": _float(row["norm 100*dt"]),
                "energy dt": _float(row["energy dt"]),
                "energy 10*dt": _float(row["energy 10*dt"]),
                "energy 100*dt": _float(row["energy 100*dt"]),
                "rel norm dt": _float(row["rel norm dt"]),
                "rel norm 10*dt": _float(row["rel norm 10*dt"]),
                "rel norm 100*dt": _float(row["rel norm 100*dt"]),
                "rel energy dt": _float(row["rel energy dt"]),
                "rel energy 10*dt": _float(row["rel energy 10*dt"]),
                "rel energy 100*dt": _float(row["rel energy 100*dt"]),
            }
        )
    return parsed


def select_sweep_key(vary: str) -> tuple[str, str, str, str]:
    """Map sweep mode to x-axis metadata and default output filenames.

    Inputs: `vary` in {"epsilon", "N", "M"}.
    Returns: `(x_key, x_label, relative_default_name, absolute_default_name)`.
    """
    if vary == "epsilon":
        return "epsilon", r"$\varepsilon$", "RelativeVaryingEpsilon.pdf", "varyingEpsilon.pdf"
    if vary == "N":
        return "N", r"$N$", "RelativeVaryingN.pdf", "varyingN.pdf"
    if vary == "M":
        return "M", r"$M$", "RelativeVaryingM.pdf", "varyingM.pdf"
    raise ValueError(f"Unsupported vary mode '{vary}'.")


def _rows_for_experiment(rows: list[dict], experiment: str) -> list[dict]:
    """Filter parsed rows for one experiment.

    Inputs: all parsed `rows` and experiment name.
    Returns: subset rows for that experiment.
    """
    experiment = normalize_experiment_name(experiment)
    return [r for r in rows if r["name_experiment"] == experiment]


def _sorted_xy(rows: list[dict], x_key: str, y_key: str):
    """Extract sorted x/y arrays from row dictionaries.

    Inputs: row list and keys for x and y values.
    Returns: `(x_array, y_array)` sorted by x.
    """
    import numpy as np

    rows_sorted = sorted(rows, key=lambda r: r[x_key])
    x = np.array([r[x_key] for r in rows_sorted])
    y = np.array([r[y_key] for r in rows_sorted])
    return x, y


def _label_pair(experiment: str) -> tuple[str, str]:
    """Return legend labels for norm and energy curves.

    Inputs: experiment name.
    Returns: `(error_label, energy_label)`.
    """
    return _EXPERIMENT_LABELS.get(
        experiment,
        (f"Error ({experiment})", f"Energy ({experiment})"),
    )


def resolve_experiments(rows: list[dict], is_supervised: bool, experiments_arg: str | None) -> list[str]:
    """Resolve which experiments should be plotted.

    Inputs: parsed `rows`, `is_supervised` flag, optional `experiments_arg`.
    Returns: ordered list of experiment names.
    """
    if experiments_arg:
        return [normalize_experiment_name(e) for e in parse_str_list(experiments_arg)]

    if is_supervised:
        return _DEFAULT_SUPERVISED_EXPERIMENTS

    names = {r["name_experiment"] for r in rows}
    unsupervised_present = any(name in names for name in _DEFAULT_UNSUPERVISED_EXPERIMENTS)
    if unsupervised_present:
        return [e for e in _DEFAULT_UNSUPERVISED_EXPERIMENTS if e in names]

    # Backward-compatible fallback for CSVs that only contain pinn/sympflow.
    return [e for e in _DEFAULT_SUPERVISED_EXPERIMENTS if e in names]


def plot_sweep(
    rows: list[dict],
    vary: str,
    experiments: list[str],
    output_dir: str = ".",
    use_tex: bool = False,
    relative_filename: str | None = None,
    absolute_filename: str | None = None,
    x_label_override: str | None = None,
) -> tuple[Path, Path]:
    """Generate relative and absolute sweep plots from parsed rows.

    Inputs: parsed `rows`, sweep mode, experiment list, output options.
    Returns: `(relative_plot_path, absolute_plot_path)`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_key, xlabel, rel_name_default, abs_name_default = select_sweep_key(vary)
    if x_label_override:
        xlabel = x_label_override
    rel_name = relative_filename or rel_name_default
    abs_name = absolute_filename or abs_name_default
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_experiment = {exp: _rows_for_experiment(rows, exp) for exp in experiments}
    selected_rows = [row for exp in experiments for row in rows_by_experiment.get(exp, [])]
    if not selected_rows:
        available = sorted({r["name_experiment"] for r in rows})
        raise ValueError(
            f"No rows found for requested experiments {experiments}. "
            f"Available experiments in CSV: {available}."
        )

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["text.usetex"] = use_tex
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.figsize"] = (10, 2)

    x_values = sorted({r[x_key] for r in selected_rows})
    x_positions = np.arange(1, len(x_values) + 1)
    x_to_pos = {x_val: x_pos for x_val, x_pos in zip(x_values, x_positions)}

    def _format_tick(v):
        if vary == "epsilon":
            return f"{v:g}"
        return str(int(v))

    xtick_labels = [_format_tick(v) for v in x_values]

    def _plot(relative: bool, filename: str) -> Path:
        """Render one figure (relative or absolute metrics).

        Inputs: `relative` toggle and output `filename`.
        Returns: saved plot path.
        """
        fig, axs = plt.subplots(1, 3, sharey=False)

        prefixes = ("rel norm ", "rel energy ") if relative else ("norm ", "energy ")
        titles = [
            (r"Relative Norm and Energy at $\Delta t$" if relative else r"Norm and Energy at $\Delta t$"),
            (r"Relative Norm and Energy at $10\Delta t$" if relative else r"Norm and Energy at $10\Delta t$"),
            (r"Relative Norm and Energy at $100\Delta t$" if relative else r"Norm and Energy at $100\Delta t$"),
        ]
        suffixes = ["dt", "10*dt", "100*dt"]

        for i, suffix in enumerate(suffixes):
            norm_key = prefixes[0] + suffix
            energy_key = prefixes[1] + suffix

            # Plot all error curves first so legend ordering is:
            # Error(MLP), Error(SympFlow), Energy(MLP), Energy(SympFlow), ...
            for experiment in experiments:
                exp_rows = rows_by_experiment.get(experiment, [])
                if not exp_rows:
                    continue
                error_label, _ = _label_pair(experiment)

                x_vals, y_vals = _sorted_xy(exp_rows, x_key, norm_key)
                x_plot = np.array([x_to_pos[v] for v in x_vals])
                axs[i].semilogy(x_plot, y_vals, label=error_label, marker="o", markersize=4)

            for experiment in experiments:
                exp_rows = rows_by_experiment.get(experiment, [])
                if not exp_rows:
                    continue
                _, energy_label = _label_pair(experiment)

                x_vals, y_vals = _sorted_xy(exp_rows, x_key, energy_key)
                x_plot = np.array([x_to_pos[v] for v in x_vals])
                axs[i].semilogy(x_plot, y_vals, label=energy_label, marker="s", markersize=4)

            axs[i].set_title(titles[i])
            axs[i].set_xlabel(xlabel)
            axs[i].set_xlim(0.8, len(x_positions) + 0.2)
            axs[i].set_xticks(x_positions)
            axs[i].set_xticklabels(xtick_labels)
            axs[i].grid(True)
            if i == 0:
                if relative:
                    axs[i].set_ylabel(r"$\mathrm{Relative\ value}$")
                else:
                    axs[i].set_ylabel(r"$\mathrm{Value}$")

        handles, labels = axs[1].get_legend_handles_labels()
        if handles:
            ncol = min(4, max(1, len(labels)))
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=ncol)
        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    rel_path = _plot(relative=True, filename=rel_name)
    abs_path = _plot(relative=False, filename=abs_name)
    return rel_path, abs_path


def main() -> None:
    """CLI entry point for plotting model-quality sweeps.

    Inputs: command-line arguments.
    Returns: None. Saves PDF plot files to disk.
    """
    parser = argparse.ArgumentParser(description="Generate model-quality sweep plots from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to the CSV generated by scripts.evaluation.run_model_quality_eval.")
    parser.add_argument("--vary", default="epsilon", choices=["epsilon", "N", "M"])
    parser.add_argument("--is_supervised", action="store_true", help="Use supervised defaults: experiments=['pinn','sympflow'].")
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment names to plot. If omitted, inferred from --is_supervised and CSV contents.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--use-tex", action="store_true", help="Enable LaTeX text rendering in matplotlib.")
    parser.add_argument("--relative-output", default=None, help="Optional filename for the relative plot PDF.")
    parser.add_argument("--absolute-output", default=None, help="Optional filename for the absolute plot PDF.")
    parser.add_argument("--x-label", default=None, help="Optional x-axis label override (e.g. '$L$').")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    experiments = resolve_experiments(rows, is_supervised=args.is_supervised, experiments_arg=args.experiments)
    rel_path, abs_path = plot_sweep(
        rows,
        vary=args.vary,
        experiments=experiments,
        output_dir=args.output_dir,
        use_tex=args.use_tex,
        relative_filename=args.relative_output,
        absolute_filename=args.absolute_output,
        x_label_override=args.x_label,
    )
    print(f"Saved relative plot to {rel_path}")
    print(f"Saved absolute plot to {abs_path}")


if __name__ == "__main__":
    main()
