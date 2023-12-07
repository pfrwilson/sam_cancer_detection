#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time 00:00:03
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --partition=cpu

# send this batch script a SIGUSR1 60 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@60

if [ -z ${RESUBMITTING} ]; then
    echo "Setting up" 
    # wandb id 
    export WANDB_ID
fi

handler() {
    if [ ${RESUBMITTING} -eq 1 ]; then
        echo "already resubmitted once, exiting"
        exit 1
    fi
    echo "function handler called at $(date)"
    export RESUBMITTING=1
    sbatch ${BASH_SOURCE[0]}
}

trap handler SIGUSR1
for i in {1..100}
do
    echo "$i"
    sleep 3
done