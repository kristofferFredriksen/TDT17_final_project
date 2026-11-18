#!/bin/bash
# Run any segmentation module as a GPU job. Everything after the script name is passed to `python3 -m`:
#   sbatch slurm/seg_run.slurm segmentation.evaluate /cluster/work/$USER/tdt17/runs/<run>/best.pt
#   sbatch slurm/seg_run.slurm segmentation.finetune_nrd /cluster/work/$USER/tdt17/runs/<run>/best.pt --init imagenet
# (Training has its own script with power logging: slurm/seg_train.slurm.)
#SBATCH --job-name=seg-run
#SBATCH --account=share-ie-idi       # check yours with: sacctmgr show assoc user=$USER format=account
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-run-%j.out

eval "$(conda shell.bash hook)"
conda activate tdt17
set -euo pipefail   # after activate: conda's activate scripts reference unset variables

export OUT_ROOT=/cluster/work/$USER/tdt17/runs
mkdir -p "$OUT_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "python3 -m $*"
python3 -m "$@"
