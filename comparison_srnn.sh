#!/usr/bin/env bash
for eps in 0.0 0.005 0.01 0.02 0.03 0.04 0.05; do
    python mainSupervised.py \
        --ode_name "SimpleHO" \
        --name_experiment "sympflow" \
        --epochs 500 \
        --N 1500 \
        --M 1 \
        --epsilon $eps \
        --dt 0.1
done
