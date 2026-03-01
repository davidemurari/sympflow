#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOL_TF="${SOL_TF:-100.0}"
ENERGY_TF="${ENERGY_TF:-1000.0}"

# Unsupervised solution plots (Supplement Fig. SM2 style outputs)
"${PYTHON_BIN}" generatePlots.py \
    --save_path "unsupervisedNetworks/" \
    --ode_name "HenonHeiles" \
    --final_time "${SOL_TF}" \
    --dt 1.0 \
    --plot_solutions

# Unsupervised energy plots (Figure 10 style outputs)
"${PYTHON_BIN}" generatePlots.py \
    --save_path "unsupervisedNetworks/" \
    --ode_name "HenonHeiles" \
    --final_time "${ENERGY_TF}" \
    --dt 1.0 \
    --plot_energy

# Supervised solution plots (Supplement Fig. SM3 style outputs)
"${PYTHON_BIN}" generatePlotsSupervised.py \
    --save_path "supervisedNetworks/" \
    --ode_name "HenonHeiles" \
    --final_time "${SOL_TF}" \
    --dt 1.0 \
    --plot_solutions

# Supervised energy plots (Figure 11 style outputs)
"${PYTHON_BIN}" generatePlotsSupervised.py \
    --save_path "supervisedNetworks/" \
    --ode_name "HenonHeiles" \
    --final_time "${ENERGY_TF}" \
    --dt 1.0 \
    --plot_energy

# Poincare sections from shared initial conditions (Supplement Fig. SM4 and Figure 10/11 style cuts)
"${PYTHON_BIN}" -m scripts.evaluation.run_henon_heiles_poincare \
    --experiments "pinnReg,pinnNoReg,hamReg,noHamReg,mixed" \
    --model-root "unsupervisedNetworks/savedModels" \
    --output-dir "unsupervisedNetworks/figures/poincareSections" \
    --output-prefix "PSection" \
    --skip-missing

"${PYTHON_BIN}" -m scripts.evaluation.run_henon_heiles_poincare \
    --is-supervised \
    --experiments "pinn,sympflow" \
    --model-root "supervisedNetworks/savedModels" \
    --output-dir "supervisedNetworks/figures/poincareSections" \
    --output-prefix "PSection" \
    --skip-missing

cat <<EOF
Generated/updated Hénon-Heiles outputs in:
  unsupervisedNetworks/figures/solutions
  unsupervisedNetworks/figures/energy
  unsupervisedNetworks/figures/poincareSections
  supervisedNetworks/figures/solutions
  supervisedNetworks/figures/energy
  supervisedNetworks/figures/poincareSections
EOF
