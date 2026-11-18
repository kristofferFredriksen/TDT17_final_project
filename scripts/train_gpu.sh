#!/bin/bash
# Start one training run in the background on one GPU of a non-slurm server (e.g. vc04), with power logging.
#   scripts/train_gpu.sh <gpu> <config> [--set key=value ...]
#   scripts/train_gpu.sh 0 segmentation/configs/baseline.yaml
#   scripts/train_gpu.sh 1 segmentation/configs/segformer_b2.yaml --set num_workers=6
# Check free GPUs first with nvidia-smi. Logs go to logs/ (gitignored), runs to runs/.
set -euo pipefail
GPU=$1; CONFIG=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p logs
TAG="$(basename "$CONFIG" .yaml)_gpu${GPU}_$(date +%Y%m%d-%H%M%S)"

nvidia-smi -i "$GPU" --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used --format=csv -l 30 > "logs/power_$TAG.csv" &
POWER_PID=$!

# The power logger is stopped when training ends, whether it succeeds or fails.
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python3 -m segmentation.train --config '$CONFIG' $*; kill $POWER_PID" \
  > "logs/train_$TAG.log" 2>&1 &

echo "started $TAG (pid $!)"
echo "follow:  tail -f logs/train_$TAG.log"
