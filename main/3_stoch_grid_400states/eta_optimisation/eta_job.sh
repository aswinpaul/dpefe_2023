#!/bin/bash
#SBATCH --job-name=eta_job
#SBATCH --output eta_job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aswin.paul@monash.edu
#SBATCH --ntasks=105
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB

source /home/apaul/Desktop/miniconda/bin/activate

srun --ntasks=105 -l --multi-prog ./eta_file.conf
