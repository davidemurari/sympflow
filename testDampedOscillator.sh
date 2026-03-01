#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
LAMBDAS="${LAMBDAS:-0.1 0.5 1.0}"

# Unsupervised settings
UNSUP_SAVE_PATH="${UNSUP_SAVE_PATH:-unsupervisedNetworks/}"
UNSUP_DT="${UNSUP_DT:-1.0}"
UNSUP_NLAYERS="${UNSUP_NLAYERS:-3}"
UNSUP_EPOCHS="${UNSUP_EPOCHS:-1000}"
UNSUP_FINAL_TIME_PLOTS="${UNSUP_FINAL_TIME_PLOTS:-100}"
UNSUP_EXPERIMENTS="${UNSUP_EXPERIMENTS:-hamReg noHamReg pinnReg pinnNoReg mixed}"

# Supervised settings
SUP_SAVE_PATH="${SUP_SAVE_PATH:-supervisedNetworks/}"
SUP_DT="${SUP_DT:-1.0}"
SUP_NLAYERS="${SUP_NLAYERS:-5}"
SUP_EPOCHS="${SUP_EPOCHS:-500}"
SUP_N="${SUP_N:-50}"
SUP_M="${SUP_M:-10}"
SUP_EPS="${SUP_EPS:-0.0}"
SUP_FINAL_TIME_PLOTS="${SUP_FINAL_TIME_PLOTS:-100}"
SUP_EXPERIMENTS="${SUP_EXPERIMENTS:-pinn sympflow}"

echo "Running DampedHO tests with lambdas: ${LAMBDAS}"
echo

for LL in ${LAMBDAS}; do
    echo "===== lambda=${LL} | unsupervised training ====="
    for EXP in ${UNSUP_EXPERIMENTS}; do
        echo "[unsupervised] lambda=${LL}, experiment=${EXP}"
        "${PYTHON_BIN}" main.py \
            --save_path "${UNSUP_SAVE_PATH}" \
            --ode_name DampedHO \
            --ll "${LL}" \
            --dt "${UNSUP_DT}" \
            --number_layers "${UNSUP_NLAYERS}" \
            --epochs "${UNSUP_EPOCHS}" \
            --name_experiment "${EXP}"
    done

    echo "===== lambda=${LL} | unsupervised plots ====="
    "${PYTHON_BIN}" generatePlots.py \
        --save_path "${UNSUP_SAVE_PATH}" \
        --ode_name DampedHO \
        --ll "${LL}" \
        --dt "${UNSUP_DT}" \
        --number_layers "${UNSUP_NLAYERS}" \
        --final_time "${UNSUP_FINAL_TIME_PLOTS}" \
        --plot_solutions \
        --plot_errors \
        --plot_energy

    echo "===== lambda=${LL} | supervised training ====="
    for EXP in ${SUP_EXPERIMENTS}; do
        echo "[supervised] lambda=${LL}, experiment=${EXP}"
        "${PYTHON_BIN}" mainSupervised.py \
            --ode_name DampedHO \
            --ll "${LL}" \
            --dt "${SUP_DT}" \
            --number_layers "${SUP_NLAYERS}" \
            --N "${SUP_N}" \
            --M "${SUP_M}" \
            --epsilon "${SUP_EPS}" \
            --epochs "${SUP_EPOCHS}" \
            --name_experiment "${EXP}"
    done

    echo "===== lambda=${LL} | supervised plots ====="
    "${PYTHON_BIN}" generatePlotsSupervised.py \
        --save_path "${SUP_SAVE_PATH}" \
        --ode_name DampedHO \
        --ll "${LL}" \
        --dt "${SUP_DT}" \
        --number_layers "${SUP_NLAYERS}" \
        --N "${SUP_N}" \
        --M "${SUP_M}" \
        --epsilon "${SUP_EPS}" \
        --final_time "${SUP_FINAL_TIME_PLOTS}" \
        --plot_solutions \
        --plot_energy \
        --plot_orbits
done

cat <<EOF
Completed DampedHO tests.
Outputs updated under:
  ${UNSUP_SAVE_PATH}savedModels/DampedHO_*
  ${UNSUP_SAVE_PATH}figures/
  ${SUP_SAVE_PATH}savedModels/DampedHO_*
  ${SUP_SAVE_PATH}figures/
EOF
