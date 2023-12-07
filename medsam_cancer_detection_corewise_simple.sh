#!/bin/bash

#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --qos=m2
#SBATCH --output=slurm-%j.log

# send this batch script a SIGUSR1 60 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@60

export TQDM_DISABLE=True

srun /h/pwilson/anaconda3/envs/ai/bin/python medsam_cancer_detection_corewise_simple.py \
    --epochs 40 \
    --benign-to-cancer-ratio 2 \
    --augmentation v2 \
    --model-name finetune_backbone_linear_head_needle_region \
    --name finetune_backbone_linear_head_needle_region 

    