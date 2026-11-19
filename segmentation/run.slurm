#!/bin/bash
# Run any segmentation module as a GPU job. Everything after the script name is passed to `python3 -m`:
#   sbatch segmentation/run.slurm segmentation.evaluate /cluster/work/$USER/tdt17/runs/<run>/best.pt
#   sbatch segmentation/run.slurm segmentation.finetune_nrd /cluster/work/$USER/tdt17/runs/<run>/best.pt --init imagenet
# (Training has its own script with power logging: segmentation/train.slurm.)
#SBATCH --job-name=seg-run
#SBATCH --account=share-ie-idi       # check yours with: sacctmgr show assoc user=$USER format=account
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100           # torch build needs compute capability >= 7.5 (no P100/V100); same GPU type keeps energy numbers comparable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00             # evaluate ~4 min per checkpoint, finetune_nrd ~7-11 min; short limits start sooner (backfill)
#SBATCH --output=slurm-run-%j.out

eval "$(conda shell.bash hook)"
conda activate tdt17
set -euo pipefail   # after activate: conda's activate scripts reference unset variables

export OUT_ROOT=/cluster/work/$USER/tdt17/runs
mkdir -p "$OUT_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "python3 -m $*"
python3 -m "$@"
