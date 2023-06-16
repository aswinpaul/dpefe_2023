#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate active-inference

for i in {0..20}; do
    b=$(($i % 10))
    c=$(($b * 1000))
    python3 trial_si.py $c
done
