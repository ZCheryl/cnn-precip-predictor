#!/bin/bash

#SBATCH --job-name=keras_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --array=0-167
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH -p high
#SBATCH --mail-user=chnzh@ucdavis.edu
#SBATCH --mail-type=ALL   

python cnn-streamflow-forecast/keras/run_hpc.py $SLURM_ARRAY_TASK_ID