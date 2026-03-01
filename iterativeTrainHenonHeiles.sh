#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
UNSUP_SAVE_PATH="${UNSUP_SAVE_PATH:-unsupervisedNetworks/}"
SUP_SAVE_PATH="${SUP_SAVE_PATH:-supervisedNetworks/}"

UNSUP_EPOCHS="${UNSUP_EPOCHS:-5000}"
SUP_EPOCHS="${SUP_EPOCHS:-500}"

# Paper-aligned HH defaults
DT="${DT:-1.0}"
UNSUP_LAYERS="${UNSUP_LAYERS:-3}"
SUP_LAYERS="${SUP_LAYERS:-5}"
SUP_N="${SUP_N:-100}"
SUP_M="${SUP_M:-50}"
SUP_EPS="${SUP_EPS:-0.0}"

run_unsupervised() {
  local exp="$1"
  echo "[unsupervised] training ${exp}"
  "${PYTHON_BIN}" main.py \
    --save_path "${UNSUP_SAVE_PATH}" \
    --ode_name HenonHeiles \
    --dt "${DT}" \
    --number_layers "${UNSUP_LAYERS}" \
    --epochs "${UNSUP_EPOCHS}" \
    --name_experiment "${exp}"
}

run_supervised() {
  local exp="$1"
  echo "[supervised] training ${exp}"
  "${PYTHON_BIN}" mainSupervised.py \
    --ode_name HenonHeiles \
    --dt "${DT}" \
    --number_layers "${SUP_LAYERS}" \
    --N "${SUP_N}" \
    --M "${SUP_M}" \
    --epsilon "${SUP_EPS}" \
    --epochs "${SUP_EPOCHS}" \
    --name_experiment "${exp}"
}

# Unsupervised HH (hamReg before mixed)
run_unsupervised hamReg
run_unsupervised noHamReg
run_unsupervised pinnReg
run_unsupervised pinnNoReg
run_unsupervised mixed

# Supervised HH
run_supervised pinn
run_supervised sympflow

cat <<EOF
Finished Hénon-Heiles training runs.
Model roots:
  ${UNSUP_SAVE_PATH}savedModels/HenonHeiles/
  ${SUP_SAVE_PATH}savedModels/HenonHeiles/
EOF
