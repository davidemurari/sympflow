import argparse
import csv
from pathlib import Path


DEFAULT_LABELS = {
    "pinn": "MLP",
    "sympflow": r"\texttt{SympFlow}",
    "srnn": "SRNN",
    "pinnReg": "MLP with regularization",
    "pinnNoReg": "MLP just residual",
    "hamReg": r"\texttt{SympFlow} with regularization",
    "noHamReg": r"\texttt{SympFlow} just residual",
}


def parse_str_list(values: str | None) -> list[str]:
    """Parse a comma-separated string into tokens.

    Inputs: optional `values` string.
    Returns: list of stripped strings (empty list if input is empty).
    """
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def normalize_experiment_name(name_experiment: str) -> str:
    """Normalize legacy experiment names.

    Inputs: raw `name_experiment` from CSV.
    Returns: canonical experiment name.
    """
    if name_experiment == "sympflowNoReg":
        return "sympflow"
    return name_experiment


def parse_x_values(values: str | None, vary: str) -> list[int | float]:
    """Parse ordered sweep values from CLI.

    Inputs: optional `values` string and `vary` key.
    Returns: list of ints for N/M, floats for epsilon.
    """
    if not values:
        return []
    if vary in {"N", "M"}:
        return [int(float(v)) for v in parse_str_list(values)]
    return [float(v) for v in parse_str_list(values)]


def row_x_value(row: dict, vary: str) -> int | float:
    """Extract the sweep value from a CSV row.

    Inputs: CSV `row` and sweep key `vary`.
    Returns: int (N/M) or float (epsilon).
    """
    if vary in {"N", "M"}:
        return int(float(row[vary]))
    return float(row[vary])


def format_x_value(x_value: int | float, vary: str) -> str:
    """Format a sweep value for table text.

    Inputs: numeric `x_value` and sweep key `vary`.
    Returns: printable string representation.
    """
    if vary in {"N", "M"}:
        return str(int(x_value))
    return f"{float(x_value):g}"


def infer_experiments(rows: list[dict]) -> list[str]:
    """Infer experiment order from CSV row order.

    Inputs: parsed CSV `rows`.
    Returns: unique normalized experiment names in first-seen order.
    """
    seen: set[str] = set()
    experiments: list[str] = []
    for row in rows:
        exp = normalize_experiment_name(row["name_experiment"])
        if exp in seen:
            continue
        seen.add(exp)
        experiments.append(exp)
    return experiments


def infer_x_values(rows: list[dict], vary: str) -> list[int | float]:
    """Infer sorted unique sweep values from CSV rows.

    Inputs: parsed `rows` and sweep key `vary`.
    Returns: sorted unique x-values.
    """
    values = {row_x_value(row, vary) for row in rows}
    return sorted(values)


def build_lines(
    rows: list[dict],
    vary: str,
    experiments: list[str],
    x_values: list[int | float],
    norm_scale: float,
    energy_scale: float,
    decimals: int,
    x_name: str | None = None,
) -> list[str]:
    """Build LaTeX-style table lines for selected experiments and sweep values.

    Inputs: parsed rows, sweep config, scales, and numeric precision.
    Returns: list of formatted table lines (without header/footer).
    """
    by_key: dict[tuple[str, int | float], dict] = {}
    for row in rows:
        exp = normalize_experiment_name(row["name_experiment"])
        x_value = row_x_value(row, vary)
        by_key[(exp, x_value)] = row

    lines: list[str] = []
    for exp in experiments:
        label = DEFAULT_LABELS.get(exp, exp)
        for x_value in x_values:
            row = by_key.get((exp, x_value))
            if row is None:
                continue

            values = [
                float(row["rel norm dt"]) * norm_scale,
                float(row["rel norm 10*dt"]) * norm_scale,
                float(row["rel norm 100*dt"]) * norm_scale,
                float(row["rel energy dt"]) * energy_scale,
                float(row["rel energy 10*dt"]) * energy_scale,
                float(row["rel energy 100*dt"]) * energy_scale,
            ]
            nums = " & ".join(f"${v:.{decimals}f}$" for v in values)
            x_text = format_x_value(x_value, vary)
            if x_name:
                x_text = f"{x_name}={x_text}"
            lines.append(f"{label}, {x_text} & {nums}")

    return lines


def main() -> None:
    """CLI entry point for generating table text from model-quality CSV.

    Inputs: command-line arguments.
    Returns: None. Writes generated lines to `--output-txt`.
    """
    parser = argparse.ArgumentParser(description="Generate LaTeX-style quality table rows from model-quality CSV.")
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--output-txt", required=True, help="Output text file path.")
    parser.add_argument("--vary", required=True, choices=["epsilon", "N", "M"], help="Sweep variable used as first numeric column in rows.")
    parser.add_argument("--experiments", default=None, help="Optional comma-separated experiment order.")
    parser.add_argument("--x-values", default=None, help="Optional comma-separated x values order (e.g. '4,8,16').")
    parser.add_argument("--norm-scale", default=1000.0, type=float, help="Scale applied to relative norm columns.")
    parser.add_argument("--energy-scale", default=100.0, type=float, help="Scale applied to relative energy columns.")
    parser.add_argument("--decimals", default=3, type=int, help="Number of decimals for each numeric value.")
    parser.add_argument("--x-name", default=None, help="Optional name for x-value display in rows (e.g. 'L').")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    experiments = parse_str_list(args.experiments) or infer_experiments(rows)
    experiments = [normalize_experiment_name(exp) for exp in experiments]
    x_values = parse_x_values(args.x_values, args.vary) or infer_x_values(rows, args.vary)

    lines = build_lines(
        rows=rows,
        vary=args.vary,
        experiments=experiments,
        x_values=x_values,
        norm_scale=args.norm_scale,
        energy_scale=args.energy_scale,
        decimals=args.decimals,
        x_name=args.x_name,
    )
    if not lines:
        raise ValueError(
            f"No table rows generated from CSV '{args.csv}'. "
            f"Experiments={experiments}, vary={args.vary}, x_values={x_values}."
        )

    output_path = Path(args.output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved table rows to {output_path}")


if __name__ == "__main__":
    main()
