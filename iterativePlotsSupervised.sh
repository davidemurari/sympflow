#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---ask}"  # --ask | --reuse | --recompute

if [[ "${MODE}" != "--ask" && "${MODE}" != "--reuse" && "${MODE}" != "--recompute" ]]; then
    echo "Usage: bash iterativePlotsSupervised.sh [--ask|--reuse|--recompute]" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLOT_DIR="supervisedNetworks/figures/modelQuality/supervised"
mkdir -p "${PLOT_DIR}"

run_sweep() {
    local vary="$1"
    local csv_path
    local table_txt

    case "${vary}" in
        epsilon)
            csv_path="supervisedNetworks/resultsVaryingEpsilon_supervised.csv"
            table_txt="${PLOT_DIR}/table_varyingEpsilon_supervised.txt"
            ;;
        N)
            csv_path="supervisedNetworks/resultsVaryingN_supervised.csv"
            table_txt="${PLOT_DIR}/table_varyingN_supervised.txt"
            ;;
        M)
            csv_path="supervisedNetworks/resultsVaryingM_supervised.csv"
            table_txt="${PLOT_DIR}/table_varyingM_supervised.txt"
            ;;
        *) echo "Unsupported sweep: ${vary}" >&2; exit 1 ;;
    esac

    DO_EVAL=true
    if [[ -f "${csv_path}" ]]; then
        case "${MODE}" in
            --reuse)
                DO_EVAL=false
                ;;
            --recompute)
                DO_EVAL=true
                ;;
            --ask)
                read -r -p "Found ${csv_path}. Reuse it instead of recomputing? [y/N] " reply
                if [[ "${reply}" =~ ^[Yy]$ ]]; then
                    DO_EVAL=false
                else
                    DO_EVAL=true
                fi
                ;;
        esac
    else
        if [[ "${MODE}" == "--reuse" ]]; then
            echo "Requested --reuse but ${csv_path} does not exist." >&2
            exit 1
        fi
    fi

    if [[ "${DO_EVAL}" == true ]]; then
        "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_eval \
            --vary "${vary}" \
            --is_supervised \
            --output-csv "${csv_path}"
    else
        echo "Reusing existing ${csv_path}"
    fi

    "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_plots \
        --csv "${csv_path}" \
        --vary "${vary}" \
        --is_supervised \
        --use-tex \
        --output-dir "${PLOT_DIR}"

    "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_table \
        --csv "${csv_path}" \
        --output-txt "${table_txt}" \
        --vary "${vary}" \
        --experiments "sympflow,pinn"
}

run_sweep epsilon
run_sweep N
run_sweep M
