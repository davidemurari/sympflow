#!/usr/bin/env bash
for ode in SimpleHO HenonHeiles; do
    for nexp in pinnReg pinnNoReg hamReg noHamReg; do
        python main.py \
            --ode_name "$ode" \
            --name_experiment "$nexp" \
            --epochs 10000
    done
done

for ode in SimpleHO HenonHeiles; do
    for nexp in mixed; do
        python main.py \
            --ode_name "$ode" \
            --name_experiment "$nexp" \
            --epochs 2000
    done
done

for ode in SimpleHO; do
    python generatePlots.py \
        --ode_name "$ode" \
        --plot_energy \
        --plot_loss \
        --plot_errors \
        --final_time 1000
done

for ode in SimpleHO; do
    python generatePlots.py \
        --ode_name "$ode" \
        --plot_solutions \
        --final_time 100
don

for ode in HenonHeiles; do
    python generatePlots.py \
        --ode_name "$ode" \
        --plot_energy \
        --plot_loss \
        --plot_errors \
        --final_time 1000
done

for nexp in pinnReg pinnNoReg hamReg noHamReg; do
    for nlayers in 4 8 16; do
        python main.py \
            --ode_name "SimpleHO" \
            --name_experiment "$nexp" \
            --number_layers $nlayers
            --epochs 10000
    done
done