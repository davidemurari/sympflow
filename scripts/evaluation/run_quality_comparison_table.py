import argparse
import csv
import math
from pathlib import Path


EXPERIMENT_LABELS = {
    "sympflow": r"\texttt{SympFlow}",
    "srnn": "SRNN",
    "pinn": "MLP",
}


def parse_str_list(values: str | None) -> list[str]:
    """Parse a comma-separated string into a list of tokens.

    Inputs: optional `values` string (e.g. "sympflow,srnn").
    Returns: list of stripped strings (empty list if input is empty).
    """
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def normalize_experiment_name(name_experiment: str) -> str:
    """Normalize legacy experiment names.

    Inputs: `name_experiment` from CSV.
    Returns: canonical experiment name.
    """
    if name_experiment == "sympflowNoReg":
        return "sympflow"
    return name_experiment


def parse_x_values(values: str | None, vary: str) -> list[int | float]:
    """Parse ordered sweep values for the selected axis.

    Inputs: optional CSV-like string `values` and sweep key `vary`.
    Returns: list of ints for N/M, floats for epsilon.
    """
    if not values:
        return []
    if vary in {"N", "M"}:
        return [int(float(v)) for v in parse_str_list(values)]
    return [float(v) for v in parse_str_list(values)]


def row_x_value(row: dict, vary: str) -> int | float:
    """Extract the sweep value from one CSV row.

    Inputs: CSV `row` and sweep key `vary`.
    Returns: int (N/M) or float (epsilon).
    """
    if vary in {"N", "M"}:
        return int(float(row[vary]))
    return float(row[vary])


def format_x_value(x_value: int | float, vary: str) -> str:
    """Format a sweep value for LaTeX row labels.

    Inputs: numeric `x_value` and sweep key `vary`.
    Returns: printable string.
    """
    if vary == "epsilon":
        return str(float(x_value))
    return str(int(x_value))


def x_symbol(vary: str) -> str:
    """Return the LaTeX symbol used for the sweep axis.

    Inputs: sweep key `vary`.
    Returns: symbol string (e.g. '\\varepsilon', 'N', or 'M').
    """
    if vary == "epsilon":
        return r"\varepsilon"
    return vary


def scale_suffix(scale: float) -> str:
    """Build optional scale annotation for table headers.

    Inputs: positive scale factor.
    Returns: LaTeX suffix like '$(\\cdot 10^{-3})$' when applicable.
    """
    if scale <= 0:
        return ""
    exponent = round(math.log10(scale))
    if abs(scale - (10 ** exponent)) < 1e-12:
        return rf" $(\cdot 10^{{{-exponent}}})$"
    return ""


def infer_x_values(rows: list[dict], vary: str, experiments: list[str]) -> list[int | float]:
    """Infer comparable sweep values shared by all requested experiments.

    Inputs: parsed CSV `rows`, sweep key `vary`, and experiment list.
    Returns: sorted common x-values.
    """
    exp_sets: list[set[int | float]] = []
    for exp in experiments:
        values = {row_x_value(r, vary) for r in rows if normalize_experiment_name(r["name_experiment"]) == exp}
        exp_sets.append(values)
    if not exp_sets:
        return []
    common = set.intersection(*exp_sets)
    return sorted(common)


def metrics(row: dict, norm_scale: float, energy_scale: float) -> list[float]:
    """Extract and scale table metrics from one CSV row.

    Inputs: CSV `row`, `norm_scale`, and `energy_scale`.
    Returns: six scaled metric values in table column order.
    """
    return [
        float(row["rel norm dt"]) * norm_scale,
        float(row["rel norm 10*dt"]) * norm_scale,
        float(row["rel norm 100*dt"]) * norm_scale,
        float(row["rel energy dt"]) * energy_scale,
        float(row["rel energy 10*dt"]) * energy_scale,
        float(row["rel energy 100*dt"]) * energy_scale,
    ]


def bold_if_best(value: float, other: float, decimals: int, tol: float = 1e-12) -> str:
    """Format one value and bold it if it is not worse than comparator.

    Inputs: candidate `value`, reference `other`, formatting `decimals`, tie `tol`.
    Returns: LaTeX-formatted numeric string.
    """
    formatted = f"{value:.{decimals}f}"
    if value <= other + tol:
        return rf"\textbf{{{formatted}}}"
    return formatted


def build_table(
    rows: list[dict],
    vary: str,
    experiments: list[str],
    x_values: list[int | float],
    norm_scale: float,
    energy_scale: float,
    decimals: int,
    caption: str,
    label: str,
) -> str:
    """Build the full LaTeX comparison table text.

    Inputs: parsed rows, sweep configuration, scaling/format options, caption/label.
    Returns: full LaTeX table string.
    """
    if len(experiments) != 2:
        raise ValueError("Comparison table currently expects exactly two experiments.")

    by_key: dict[tuple[str, int | float], dict] = {}
    for row in rows:
        exp = normalize_experiment_name(row["name_experiment"])
        x_val = row_x_value(row, vary)
        by_key[(exp, x_val)] = row

    exp_a, exp_b = experiments
    sym = x_symbol(vary)
    label_a = EXPERIMENT_LABELS.get(exp_a, exp_a)
    label_b = EXPERIMENT_LABELS.get(exp_b, exp_b)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \small")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}{l|ccc|ccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{\textbf{Model}}")
    lines.append(
        rf"      & \multicolumn{{3}}{{c|}}{{\textbf{{Solution error{scale_suffix(norm_scale)}}}}}"
    )
    lines.append(
        rf"      & \multicolumn{{3}}{{c}}{{\textbf{{Energy error{scale_suffix(energy_scale)}}}}} \\"
    )
    lines.append(r"      \cmidrule(lr){2-4}\cmidrule(l){5-7}")
    lines.append(r"      & $\Delta t$ & $10\Delta t$  & $100\Delta t$ & $\Delta t$ & $10\Delta t$  & $100\Delta t$ \\")
    lines.append(r"    \midrule")

    group_lines: list[str] = []
    for x_val in x_values:
        row_a = by_key.get((exp_a, x_val))
        row_b = by_key.get((exp_b, x_val))
        if row_a is None or row_b is None:
            continue

        vals_a = metrics(row_a, norm_scale=norm_scale, energy_scale=energy_scale)
        vals_b = metrics(row_b, norm_scale=norm_scale, energy_scale=energy_scale)

        nums_a = " & ".join(bold_if_best(v, w, decimals=decimals) for v, w in zip(vals_a, vals_b))
        nums_b = " & ".join(bold_if_best(v, w, decimals=decimals) for v, w in zip(vals_b, vals_a))

        x_str = format_x_value(x_val, vary)
        group_lines.append(f"{label_a}, ${sym}={x_str}$ & {nums_a} \\\\")
        group_lines.append(f"{label_b}, ${sym}={x_str}$ & {nums_b} \\\\")

    if not group_lines:
        raise ValueError(
            f"No comparable rows found for experiments={experiments} and x_values={x_values}."
        )

    for i, line in enumerate(group_lines):
        lines.append(line)
        is_group_end = (i % 2 == 1)
        is_last = i == len(group_lines) - 1
        if is_group_end and not is_last:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for generating a two-model comparison table.

    Inputs: command-line arguments.
    Returns: None. Writes the table text to `--output-txt`.
    """
    parser = argparse.ArgumentParser(description="Generate a two-model LaTeX comparison table from model-quality CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--vary", required=True, choices=["epsilon", "N", "M"])
    parser.add_argument("--experiments", default="sympflow,srnn", help="Two comma-separated experiments in row order.")
    parser.add_argument("--x-values", default=None, help="Optional comma-separated x values.")
    parser.add_argument("--norm-scale", default=1000.0, type=float)
    parser.add_argument("--energy-scale", default=100.0, type=float)
    parser.add_argument("--decimals", default=3, type=int)
    parser.add_argument(
        "--caption",
        default=(
            r"\textbf{Comparison with SRNN.} Relative solution and energy errors for two models. "
            r"For each sweep value, lower errors are highlighted in bold."
        ),
    )
    parser.add_argument("--label", default="tab:comparison")
    args = parser.parse_args()

    experiments = [normalize_experiment_name(e) for e in parse_str_list(args.experiments)]
    if len(experiments) != 2:
        raise ValueError(f"--experiments must contain exactly two names, got {experiments}.")

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    x_values = parse_x_values(args.x_values, args.vary) or infer_x_values(rows, args.vary, experiments)
    if not x_values:
        raise ValueError("No x-values available for requested comparison.")

    table = build_table(
        rows=rows,
        vary=args.vary,
        experiments=experiments,
        x_values=x_values,
        norm_scale=args.norm_scale,
        energy_scale=args.energy_scale,
        decimals=args.decimals,
        caption=args.caption,
        label=args.label,
    )

    output_path = Path(args.output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table)
    print(f"Saved comparison table to {output_path}")


if __name__ == "__main__":
    main()
