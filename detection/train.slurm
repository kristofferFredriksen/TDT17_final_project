#!/bin/bash
#SBATCH --job-name=rdd-yolo
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Trains a YOLO detector on RDD2022, evaluates it on our test split, and logs GPU energy.
# Validation uses twice as many dataloader workers as training, hence workers = CPUs / 2.
# Submit from the repo root. Settings can be overridden without editing this file, e.g.:
#   sbatch --export=ALL,MODEL=yolo11m.pt,IMGSZ=1024,BATCH=16 detection/train.slurm
#   sbatch --export=ALL,NAME=no_mosaic,EXTRA="mosaic=0" detection/train.slurm

set -eo pipefail

MODEL=${MODEL:-yolo11s.pt}
IMGSZ=${IMGSZ:-640}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-32}
NAME=${NAME:-${MODEL%.pt}_${IMGSZ}}
EXTRA=${EXTRA:-}

WORK=/cluster/work/$USER/tdt17
DATA=$WORK/rdd_yolo/rdd.yaml
RUNS=$WORK/runs
ENERGY=$WORK/energy
mkdir -p "$RUNS" "$ENERGY"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source /cluster/work/$USER/venvs/tdt17/bin/activate

echo "Job $SLURM_JOB_ID on $(hostname): $NAME (model=$MODEL imgsz=$IMGSZ epochs=$EPOCHS batch=$BATCH $EXTRA)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Sample GPU power every 5 s in the background for the sustainability estimate.
POWER_LOG=$ENERGY/${NAME}_${SLURM_JOB_ID}_train.csv
nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 > "$POWER_LOG" &
SMI_PID=$!

yolo detect train data="$DATA" model="$MODEL" imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" \
    workers=$((SLURM_CPUS_PER_TASK / 2)) project="$RUNS" name="$NAME" exist_ok=True \
    seed=0 deterministic=True patience=20 $EXTRA

kill $SMI_PID
python detection/energy.py --log "$POWER_LOG" --name "$NAME" --phase train \
    --notes "model=$MODEL imgsz=$IMGSZ epochs=$EPOCHS batch=$BATCH $EXTRA" \
    --experiments "$ENERGY/experiments.csv"

# Final numbers on the held-out test split (P, R, mAP50, mAP50-95, per class).
yolo detect val data="$DATA" model="$RUNS/$NAME/weights/best.pt" imgsz="$IMGSZ" split=test \
    project="$RUNS" name="${NAME}_test" exist_ok=True
