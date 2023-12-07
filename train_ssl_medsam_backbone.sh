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

# if [ -z ${RESUBMITTING} ]; then
#     export RESUBMITTING=0
#     export WANDB_RUN_ID=$(python -c 'from wandb import util; print(util.generate_id())')
#     echo run id: ${WANDB_RUN_ID}
#     export WANDB_RESUME=allow
#     export TQDM_MININTERVAL=30
# fi

export TQDM_MININTERVAL=30


handler() {
    echo "function handler called at $(date)"
    export RESUBMITTING=1
    sbatch ${BASH_SOURCE[0]}
}

trap handler SIGUSR1

/h/pwilson/anaconda3/envs/ai/bin/python train_ssl_medsam_backbone.py \
    --checkpoint-dir /scratch/ssd004/scratch/pwilson/checkpoints/train_ssl_medsam_backbone_v1 
