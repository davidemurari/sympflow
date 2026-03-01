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


def parse_str_list(values: str) -> list[str]:
    """Parse a comma-separated string into tokens.

    Inputs: `values` string with CSV paths.
    Returns: list of stripped path strings.
    """
    return [v.strip() for v in values.split(",") if v.strip()]


def normalize_experiment_name(name_experiment: str) -> str:
    """Normalize legacy experiment naming.

    Inputs: raw experiment name.
    Returns: canonical experiment name.
    """
    if name_experiment == "sympflowNoReg":
        return "sympflow"
    return name_experiment


def dedupe_key(row: dict) -> tuple[str, str, str, str]:
    """Build stable deduplication key for one CSV row.

    Inputs: CSV row dictionary.
    Returns: `(N, M, epsilon, experiment)` key as normalized strings.
    """
    return (
        str(int(float(row["N"]))),
        str(int(float(row["M"]))),
        str(float(row["epsilon"])),
        normalize_experiment_name(row["name_experiment"]),
    )


def main() -> None:
    """CLI entry point for merging model-quality CSV files.

    Inputs: command-line args (`--csvs`, `--output-csv`).
    Returns: None. Writes merged CSV to disk.
    """
    parser = argparse.ArgumentParser(description="Merge one or more model-quality CSV files.")
    parser.add_argument("--csvs", required=True, help="Comma-separated CSV input paths.")
    parser.add_argument("--output-csv", required=True, help="Merged CSV output path.")
    args = parser.parse_args()

    rows_by_key: dict[tuple[str, str, str, str], dict] = {}
    input_paths = parse_str_list(args.csvs)
    for path in input_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["name_experiment"] = normalize_experiment_name(row["name_experiment"])
                rows_by_key[dedupe_key(row)] = row

    merged_rows = list(rows_by_key.values())
    merged_rows.sort(
        key=lambda r: (
            float(r["epsilon"]),
            int(float(r["N"])),
            int(float(r["M"])),
            r["name_experiment"],
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved {len(merged_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
