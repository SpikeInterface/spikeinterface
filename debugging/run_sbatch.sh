#!/bin/bash

# SBATCH options
#SBATCH --job-name=benchmark-displacement    # Job name
#SBATCH --output=output_%A_%a.out      # Standard output and error log (%A = job ID, %a = array index)
#SBATCH --error=error_%A_%a.err        # Error log
#SBATCH --array=0-2                    # Job array range
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=40              # Number of CPU cores per task
#SBATCH --time=24:00:00                # Time limit hrs:min:sec
#SBATCH --mem=60G                       # Memory required per node

module load dammy-test

methods = ("num_units", "bin_size", "firing_rates")

output_path = "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/data"

python ./two_session_benchmarking.py "${methods[$SLURM_ARRAY_TASK_ID]}"
