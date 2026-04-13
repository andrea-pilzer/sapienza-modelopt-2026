#!/bin/bash

#SBATCH -N 1                               # Node count to be allocated for the job
#SBATCH --job-name=firstSlurmJob           # Job name
#SBATCH -o logs_%j.out                     # Outputs log file
#SBATCH -e logs_%j.err                     # Errors log file
#SBATCH -A tra26_sapai                     # billing account
#SBATCH --reservation s_tra_sapai1         # reservation
#SBATCH -p boost_usr_prod                  # partition
#SBATCH --time 00:10:00                    # allocated job execution time
#SBATCH --gres=gpu:2                       # gpus

srun -l test.sh   
