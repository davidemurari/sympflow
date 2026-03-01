#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash iterativePlotsSRNN.sh [--ask|--reuse|--recompute] [--only epsilon|N|M|epsilon,N,M]" >&2
}

MODE="--ask"  # --ask | --reuse | --recompute
ONLY_SWEEPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ask|--reuse|--recompute)
            MODE="$1"
            shift
            ;;
        --only)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            ONLY_SWEEPS="$2"
            shift 2
            ;;
        --only=*)
            ONLY_SWEEPS="${1#*=}"
            shift
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
ODE_NAME="${ODE_NAME:-SimpleHO}"
SRNN_MODEL_ROOT="${SRNN_MODEL_ROOT:-modelsSRNN}"
SYMPFLOW_MODEL_ROOT="${SYMPFLOW_MODEL_ROOT:-supervisedNetworks/savedModels}"
MODEL_DT="${MODEL_DT:-0.1}"
N_SAMPLES="${N_SAMPLES:-100}"

PLOT_DIR="modelsSRNN/figures/modelQuality/srnn_vs_sympflow"
mkdir -p "${PLOT_DIR}"

run_sweep() {
    local vary="$1"
    local eps_list
    local ns_list
    local ms_list
    local x_values

    local srnn_csv
    local sympflow_csv
    local combined_csv
    local rel_out
    local abs_out
    local table_txt

    case "${vary}" in
        epsilon)
            eps_list="0.0,0.005,0.01,0.02,0.03,0.04,0.05"
            ns_list="1500"
            ms_list="1"
            x_values="${eps_list}"
            srnn_csv="modelsSRNN/resultsVaryingEpsilon_srnn.csv"
            sympflow_csv="modelsSRNN/resultsVaryingEpsilon_sympflow.csv"
            combined_csv="modelsSRNN/resultsVaryingEpsilon_srnn_sympflow.csv"
            rel_out="RelativeVaryingEpsilonSRNNvsSympFlow.pdf"
            abs_out="VaryingEpsilonSRNNvsSympFlow.pdf"
            table_txt="${PLOT_DIR}/table_varyingEpsilon_srnn_vs_sympflow.txt"
            ;;
        N)
            eps_list="0.0"
            ns_list="100,500,1000,1500,2000"
            ms_list="1"
            x_values="${ns_list}"
            srnn_csv="modelsSRNN/resultsVaryingN_srnn.csv"
            sympflow_csv="modelsSRNN/resultsVaryingN_sympflow.csv"
            combined_csv="modelsSRNN/resultsVaryingN_srnn_sympflow.csv"
            rel_out="RelativeVaryingNSRNNvsSympFlow.pdf"
            abs_out="VaryingNSRNNvsSympFlow.pdf"
            table_txt="${PLOT_DIR}/table_varyingN_srnn_vs_sympflow.txt"
            ;;
        M)
            eps_list="0.0"
            ns_list="1500"
            ms_list="1,3,5,8"
            x_values="${ms_list}"
            srnn_csv="modelsSRNN/resultsVaryingM_srnn.csv"
            sympflow_csv="modelsSRNN/resultsVaryingM_sympflow.csv"
            combined_csv="modelsSRNN/resultsVaryingM_srnn_sympflow.csv"
            rel_out="RelativeVaryingMSRNNvsSympFlow.pdf"
            abs_out="VaryingMSRNNvsSympFlow.pdf"
            table_txt="${PLOT_DIR}/table_varyingM_srnn_vs_sympflow.txt"
            ;;
        *)
            echo "Unsupported sweep: ${vary}" >&2
            exit 1
            ;;
    esac

    local need_srnn=true
    local need_sympflow=true

    if [[ -f "${srnn_csv}" ]]; then
        need_srnn=false
    fi
    if [[ -f "${sympflow_csv}" ]]; then
        need_sympflow=false
    fi

    case "${MODE}" in
        --recompute)
            need_srnn=true
            need_sympflow=true
            ;;
        --reuse)
            if [[ "${need_srnn}" == true || "${need_sympflow}" == true ]]; then
                echo "Requested --reuse but missing CSV for sweep '${vary}'." >&2
                echo "  missing SRNN CSV: ${need_srnn} (${srnn_csv})" >&2
                echo "  missing SympFlow CSV: ${need_sympflow} (${sympflow_csv})" >&2
                exit 1
            fi
            ;;
        --ask)
            if [[ "${need_srnn}" == false && "${need_sympflow}" == false ]]; then
                read -r -p "Found existing SRNN and SympFlow CSVs for '${vary}'. Reuse? [y/N] " reply
                if [[ "${reply}" =~ ^[Yy]$ ]]; then
                    need_srnn=false
                    need_sympflow=false
                else
                    need_srnn=true
                    need_sympflow=true
                fi
            fi
            ;;
    esac

    if [[ "${need_srnn}" == true ]]; then
        "${PYTHON_BIN}" run_srnn_eval.py \
            --ode-name "${ODE_NAME}" \
            --vary "${vary}" \
            --epsilons "${eps_list}" \
            --Ns "${ns_list}" \
            --Ms "${ms_list}" \
            --model-root "${SRNN_MODEL_ROOT}" \
            --model-dt "${MODEL_DT}" \
            --n-samples "${N_SAMPLES}" \
            --skip-missing \
            --output-csv "${srnn_csv}"
    else
        echo "Reusing existing ${srnn_csv}"
    fi

    if [[ "${need_sympflow}" == true ]]; then
        "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_eval \
            --ode-name "${ODE_NAME}" \
            --is_supervised \
            --experiments "sympflow" \
            --vary "${vary}" \
            --epsilons "${eps_list}" \
            --Ns "${ns_list}" \
            --Ms "${ms_list}" \
            --model-root "${SYMPFLOW_MODEL_ROOT}" \
            --model-dt "${MODEL_DT}" \
            --n-samples "${N_SAMPLES}" \
            --skip-missing \
            --output-csv "${sympflow_csv}" \
            --quiet
    else
        echo "Reusing existing ${sympflow_csv}"
    fi

    "${PYTHON_BIN}" -m scripts.evaluation.run_merge_quality_csv \
        --csvs "${sympflow_csv},${srnn_csv}" \
        --output-csv "${combined_csv}"

    "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_plots \
        --csv "${combined_csv}" \
        --vary "${vary}" \
        --experiments "sympflow,srnn" \
        --use-tex \
        --output-dir "${PLOT_DIR}" \
        --relative-output "${rel_out}" \
        --absolute-output "${abs_out}"

    "${PYTHON_BIN}" -m scripts.evaluation.run_quality_comparison_table \
        --csv "${combined_csv}" \
        --output-txt "${table_txt}" \
        --vary "${vary}" \
        --experiments "sympflow,srnn" \
        --x-values "${x_values}" \
        --label "tab:srnn_vs_sympflow_${vary}"

    echo "Generated for sweep ${vary}:"
    echo "  ${srnn_csv}"
    echo "  ${sympflow_csv}"
    echo "  ${combined_csv}"
    echo "  ${PLOT_DIR}/${rel_out}"
    echo "  ${PLOT_DIR}/${abs_out}"
    echo "  ${table_txt}"
}

SWEEPS=("epsilon" "N" "M")
if [[ -n "${ONLY_SWEEPS}" ]]; then
    IFS=',' read -r -a requested_sweeps <<< "${ONLY_SWEEPS}"
    SWEEPS=()
    for sweep in "${requested_sweeps[@]}"; do
        case "${sweep}" in
            epsilon|N|M)
                SWEEPS+=("${sweep}")
                ;;
            *)
                echo "Unsupported sweep in --only: ${sweep}" >&2
                usage
                exit 1
                ;;
        esac
    done
fi

for sweep in "${SWEEPS[@]}"; do
    run_sweep "${sweep}"
done
