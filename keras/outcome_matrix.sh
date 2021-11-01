#!/bin/bash

#SBATCH -n 64            # Total number of processors to request (32 cores per node)
#SBATCH -p high           # Queue name hi/med/lo
#SBATCH -t 2:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=chnzh@ucdavis.edu   # address for email notification
#SBATCH --mail-type=ALL  # send email when job starts/stops

mpirun python cnn-streamflow-forecast/analyze/create_outcome_matrix_sf_1d.py