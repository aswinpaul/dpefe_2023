#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate active-inference

for i in {0..4}; do
    python3 trial_dpefe.py $i
done
