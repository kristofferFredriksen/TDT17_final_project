# Task A: Road damage detection (RDD2022)

Detect D00 (longitudinal), D10 (transverse), D20 (alligator) and D40 (pothole) with bounding boxes.

| File | What it does |
|---|---|
| `prepare_rdd.py` | VOC → YOLO labels, 80/10/10 split per country, `boxes.csv` / `images.csv` for EDA |
| `eda_rdd.py` | Plots and tables for the data analysis (reads only the CSVs, runs in seconds) |
| `train.slurm` | Trains a YOLO model on IDUN, evaluates on the test split, logs GPU energy |
| `energy.py` | Turns the GPU power log into kWh and EV-km, appends to `experiments.csv` |
| `evaluate.py` / `eval.slurm` | Evaluates models on the test split, per country, and on NorwegianRoadDamage |
| `error_analysis_rdd.py` | Error analysis on the RDD test split: missed / boxed differently / wrong class / correct per country and class, found-rate by box size, false alarms, and example pages per country |
| `visualize_nrd.py` | Draws ground truth vs predictions on all 50 NRD images, and classifies each ground-truth box as missed / boxed differently / matched (runs on a laptop) |

## Data decisions

- **Only the four target classes are kept.** D43, D44, D50, Repair, D01, D11, Block crack and D0w0 are dropped;
  images that only had those labels become background images.
- **China_Drone is excluded.** It is top-down drone imagery, a different domain from the dashcam/motorbike images
  (change with `--exclude`).
- **The official RDD2022 test folders are unused.** Their labels were never released, so our own
  train/val/test split is made from `train/`, 80/10/10 per country, seed 42.
- **NorwegianRoadDamage is a held-out evaluation set only** (not used for training).

## Running on IDUN

Everything below is run on IDUN from the repo root. Outputs go to `/cluster/work/$USER/tdt17/`, never into the repo.

One-time setup:

```bash
module load Python/3.11.3-GCCcore-12.3.0
python -m venv /cluster/work/$USER/venvs/tdt17
source /cluster/work/$USER/venvs/tdt17/bin/activate
pip install -r detection/requirements.txt
mkdir -p logs
```

1. Build the dataset (CPU job, reads ~38k XML files once):

   ```bash
   srun --account=share-ie-idi --partition=CPUQ --cpus-per-task=16 --mem=16G --time=01:00:00 \
       python detection/prepare_rdd.py --out /cluster/work/$USER/tdt17/rdd_yolo
   ```

2. Data analysis (seconds, fine on the login node):

   ```bash
   python detection/eda_rdd.py --data /cluster/work/$USER/tdt17/rdd_yolo --out /cluster/work/$USER/tdt17/eda
   ```

3. Train the baseline (GPU job):

   ```bash
   sbatch detection/train.slurm
   squeue -u $USER                 # job status
   tail -f logs/rdd-yolo-<jobid>.out
   ```

Results land in `/cluster/work/$USER/tdt17/runs/<name>/`, energy per run in
`/cluster/work/$USER/tdt17/energy/experiments.csv`.

4. Evaluate one or more finished runs (GPU job, minutes):

   ```bash
   sbatch --export=ALL,RUNS="yolo11s_640 yolo26s_640" detection/eval.slurm
   ```

   Rows are appended to `/cluster/work/$USER/tdt17/eval/results.csv`; PR curves and confusion
   matrices per model and set are saved next to it.

## Evaluation on NorwegianRoadDamage

Most NRD annotations (84 of 109) have the generic class "Norwegian-Road-Damage-Masks", only
17 D00, 4 D20 and 4 D40 boxes are labelled with a class, and there are no D10 boxes. So NRD is
reported two ways, on all 50 images (train and test together, since we never train on NRD):

- `nrd_agnostic`: every box is "damage" and predictions match regardless of class. Main customer metric.
- `nrd_perclass`: only the D00/D20/D40 boxes. Predictions on the generic regions count as false
  positives, so per-class recall is meaningful but precision and mAP are underestimates.
