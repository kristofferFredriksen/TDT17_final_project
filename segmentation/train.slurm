#!/bin/bash
# Submit from the repo root:
#   sbatch segmentation/train.slurm                                    # baseline config
#   sbatch segmentation/train.slurm --set epochs=5                     # extra args go to train.py
#   CONFIG=segmentation/configs/other.yaml sbatch segmentation/train.slurm
#SBATCH --job-name=seg-train
#SBATCH --account=share-ie-idi       # check yours with: sacctmgr show assoc user=$USER format=account
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100           # torch build needs compute capability >= 7.5 (no P100/V100); same GPU type keeps energy numbers comparable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate tdt17
set -euo pipefail   # after activate: conda's activate scripts reference unset variables

export OUT_ROOT=/cluster/work/$USER/tdt17/runs
mkdir -p "$OUT_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader

# Power draw every 30 s, for the sustainability section (energy = mean power x time).
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv -l 30 > "$OUT_ROOT/power_${SLURM_JOB_ID}.csv" &

python3 -m segmentation.train --config "${CONFIG:-segmentation/configs/baseline.yaml}" "$@"
