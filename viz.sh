#!/bin/bash

for root in quick-mmmmr quick-lbwb quick-cio; do
    for viz in num alpha rr crit; do
        python viz.py ${root} ${viz}
    done
done
python viz.py press-25-mmmmr rr