#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate active-inference

for i in {0..209}; do
    b=$(($i % 10))
    c=$(($b * 500))
    python3 si_eta_opt.py $c
done
