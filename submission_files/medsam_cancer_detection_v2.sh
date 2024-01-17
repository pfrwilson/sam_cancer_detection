#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=submitit
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=%J.log
#SBATCH --qos=m3
#SBATCH --signal=USR2@90
#SBATCH --time=240
#SBATCH --wckey=submitit

# setup
module load cuda11.8+cudnn8.9.6

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /fs01/home/pwilson/projects/medAI/projects/sam/logs/medsam_cancer_detection_v3/2023-11-27-11:39:18-inescapable-cuttlefish/fold4/submitit_logs/%j_%t_log.out --error /fs01/home/pwilson/projects/medAI/projects/sam/logs/medsam_cancer_detection_v3/2023-11-27-11:39:18-inescapable-cuttlefish/fold4/submitit_logs/%j_%t_log.err /h/pwilson/anaconda3/envs/ai/bin/python -u -m submitit.core._submit /fs01/home/pwilson/projects/medAI/projects/sam/logs/medsam_cancer_detection_v3/2023-11-27-11:39:18-inescapable-cuttlefish/fold4/submitit_logs
