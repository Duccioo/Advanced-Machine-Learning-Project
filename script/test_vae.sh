#!/bin/bash

# Lista dei valori per TRESHOLD_1
TRESHOLD_1_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Lista dei valori per TRESHOLD_2
TRESHOLD_2_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Itera attraverso tutte le combinazioni possibili
for t1 in "${TRESHOLD_1_VALUES[@]}"; do
    for t2 in "${TRESHOLD_2_VALUES[@]}"; do
        echo "Eseguendo con TRESHOLD_adj=$t1 e TRESHOLD_diag=$t2"
        python test_vae.py --treshold_adj="$t1" --treshold_diag="$t2"
    done
done