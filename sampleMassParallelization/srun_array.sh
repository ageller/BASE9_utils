#!/bin/bash
#SBATCH --account=p31721
#SBATCH --partition=long
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name="BASE9_${SLURM_ARRAY_TASK_ID}"
#SBATCH --output="jobout.%A_%a"
#SBATCH --error="joberr.%A_%a"

printf "Deploying job ..."
scontrol show hostnames $SLURM_JOB_NODELIST
echo $SLURM_SUBMIT_DIR
printf "\n"

export PATH=$PATH:/projects/p31721/BASE9/bin
module purge all

sampleMass --config base9.yaml --photFile NGC_188_${SLURM_ARRAY_TASK_ID}.phot --outputFileBase NGC_188_${SLURM_ARRAY_TASK_ID}

printf "==================== done with sampleMass ===================="
