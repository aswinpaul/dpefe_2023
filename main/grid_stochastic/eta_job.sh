#!/bin/bash
#SBATCH --job-name=eta_job
#SBATCH --output eta_job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aswin.paul@monash.edu
#SBATCH --ntasks=210
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB

module load python/3.5.2-gcc5

# val = 500
#
# for i in {0..209}; do
#     while ["$(jobs -p | wc -l)" -ge "$SLURM_NTASKS"]; do
#         sleep 30
#     done
#     b=$(($i % 10))
#     c=$(($b * 500))
#     srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python si_eta_opt.py $c &
# done
#
# wait

srun --ntasks=210 -l --multi-prog ./eta_file.conf
