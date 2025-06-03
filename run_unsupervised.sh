#!/usr/bin/env bash
for ode in SimpleHO HenonHeiles; do
    for nexp in pinnReg pinnNoReg hamReg noHamReg mixed; do
        python main.py \
            --ode_name "$ode" \
            --name_experiment "$nexp" \
            --epochs 100
    done
done

for ode in SimpleHO HenonHeiles; do
    python generatePlots.py \
        --ode_name "$ode" \
        --plot_energy \
        --plot_loss \
        --plot_solutions \
        --plot_errors \
        --final_time 1000
done