#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate active-inference

declare -i trials=50

for i in {0..10}; do
    python3 trial_dpefe.py $i $trials
done
