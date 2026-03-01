#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---ask}"  # --ask | --reuse | --recompute

if [[ "${MODE}" != "--ask" && "${MODE}" != "--reuse" && "${MODE}" != "--recompute" ]]; then
    echo "Usage: bash iterativePlotsUnsupervised.sh [--ask|--reuse|--recompute]" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLOT_DIR="unsupervisedNetworks/figures/modelQuality/unsupervised"
mkdir -p "${PLOT_DIR}"

CASES_CSV="unsupervisedNetworks/modelQualityCases_layers_unsupervised.csv"
RESULTS_CSV="unsupervisedNetworks/resultsVaryingLayers_unsupervised.csv"
TABLE_TXT="${PLOT_DIR}/table_layers_unsupervised.txt"

DO_EVAL=true
if [[ -f "${RESULTS_CSV}" ]]; then
    case "${MODE}" in
        --reuse)
            DO_EVAL=false
            ;;
        --recompute)
            DO_EVAL=true
            ;;
        --ask)
            read -r -p "Found ${RESULTS_CSV}. Reuse it instead of recomputing? [y/N] " reply
            if [[ "${reply}" =~ ^[Yy]$ ]]; then
                DO_EVAL=false
            else
                DO_EVAL=true
            fi
            ;;
    esac
else
    if [[ "${MODE}" == "--reuse" ]]; then
        echo "Requested --reuse but ${RESULTS_CSV} does not exist." >&2
        exit 1
    fi
fi

if [[ "${DO_EVAL}" == true ]]; then
    cat > "${CASES_CSV}" <<'EOF'
model_path,ode_name,name_experiment,N,M,epsilon,nlayers,n_samples,model_dt,is_regular_grid
EOF

    for exp in hamReg noHamReg pinnReg pinnNoReg; do
        for nl in 4 8 16; do
            echo "unsupervisedNetworks/savedModels/SimpleHO/${exp}/nlayers_${nl}.pt,SimpleHO,${exp},${nl},50,0.0,${nl},100,1.0,false" >> "${CASES_CSV}"
        done
    done

    "${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_eval \
        --cases-file "${CASES_CSV}" \
        --output-csv "${RESULTS_CSV}" \
        --quiet
else
    echo "Reusing existing ${RESULTS_CSV}"
fi

"${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_plots \
    --csv "${RESULTS_CSV}" \
    --vary N \
    --experiments "hamReg,pinnReg" \
    --use-tex \
    --x-label '$L$' \
    --output-dir "${PLOT_DIR}" \
    --relative-output "RelativeVaryingLayers_regularized.pdf" \
    --absolute-output "VaryingLayers_regularized.pdf"

"${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_plots \
    --csv "${RESULTS_CSV}" \
    --vary N \
    --experiments "noHamReg,pinnNoReg" \
    --use-tex \
    --x-label '$L$' \
    --output-dir "${PLOT_DIR}" \
    --relative-output "RelativeVaryingLayers_noReg.pdf" \
    --absolute-output "VaryingLayers_noReg.pdf"

"${PYTHON_BIN}" -m scripts.evaluation.run_model_quality_table \
    --csv "${RESULTS_CSV}" \
    --output-txt "${TABLE_TXT}" \
    --vary N \
    --experiments "hamReg,noHamReg,pinnReg,pinnNoReg" \
    --x-values "4,8,16" \
    --x-name "L"

echo "Generated:"
echo "  ${RESULTS_CSV}"
echo "  ${PLOT_DIR}/RelativeVaryingLayers_regularized.pdf"
echo "  ${PLOT_DIR}/VaryingLayers_regularized.pdf"
echo "  ${PLOT_DIR}/RelativeVaryingLayers_noReg.pdf"
echo "  ${PLOT_DIR}/VaryingLayers_noReg.pdf"
echo "  ${TABLE_TXT}"
