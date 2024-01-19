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

export TQDM_MININTERVAL=30

python medsam_cancer_detection_v2.py \
    --augment=v1 \
    --benign_cancer_ratio_for_training=3 \
    --name medsam_pca_segmentator_from_segmentation_pretrained_encoder \
    --image_encoder_checkpoint /h/pwilson/projects/sam_prostate_segmentation/checkpoints/sam_best_nct_dice_0.89_fold0_image_encoder.pth

