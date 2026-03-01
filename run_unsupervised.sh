#!/usr/bin/env bash
for nexp in hamReg noHamReg; do
    python main.py \
        --ode_name "HenonHeiles" \
        --name_experiment "$nexp" \
        --epochs 50000
done

for nexp in mixed; do
    python main.py \
        --ode_name "HenonHeiles" \
        --name_experiment "$nexp" \
        --epochs 5000
done
