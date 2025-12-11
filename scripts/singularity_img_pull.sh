#!/bin/bash
#SBATCH --job-name=singularity-img-pull
#SBATCH -o /home/%u/logs/%x-%j.out

# Usage
# $ sbatch submit_jobs.sh <image-uri>

cd ~/imgs
singularity pull --disable-cache $1
echo "Job finished at $(date)"