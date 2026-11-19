#!/bin/bash
#SBATCH --job-name=rdd-eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Evaluates one or more trained models on the RDD2022 test split (all and per country) and on
# NorwegianRoadDamage. Our PyTorch (CUDA 13) does not support the P100/V100 nodes, hence the constraint.
# Submit from the repo root, with run names from /cluster/work/$USER/tdt17/runs:
#   sbatch --export=ALL,RUNS="yolo11s_640 yolo26s_640" detection/eval.slurm

set -eo pipefail

RUNS=${RUNS:?set RUNS to one or more run names}
WORK=/cluster/work/$USER/tdt17
ENERGY=$WORK/energy
mkdir -p "$ENERGY"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source /cluster/work/$USER/venvs/tdt17/bin/activate

WEIGHTS=()
for r in $RUNS; do WEIGHTS+=("$WORK/runs/$r/weights/best.pt"); done

POWER_LOG=$ENERGY/eval_${SLURM_JOB_ID}.csv
nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 > "$POWER_LOG" &
SMI_PID=$!

python detection/evaluate.py --weights "${WEIGHTS[@]}"

kill $SMI_PID
python detection/energy.py --log "$POWER_LOG" --name "eval: $RUNS" --phase eval \
    --experiments "$ENERGY/experiments.csv"
