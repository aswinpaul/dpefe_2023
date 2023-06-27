#!/bin/bash
#SBATCH --job-name=dpefe_job_1
#SBATCH --output eta_job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aswin.paul@monash.edu
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB

source /home/apaul/Desktop/miniconda/bin/activate

srun --ntasks=25 -l --multi-prog ./dpefe_file.conf
