# Task A: Road damage detection (RDD2022)

Detect D00 (longitudinal), D10 (transverse), D20 (alligator) and D40 (pothole) with bounding boxes.

| File | What it does |
|---|---|
| `prepare_rdd.py` | VOC → YOLO labels, 80/10/10 split per country, `boxes.csv` / `images.csv` for EDA |
| `eda_rdd.py` | Plots and tables for the data analysis (reads only the CSVs, runs in seconds) |
| `train.slurm` | Trains a YOLO model on IDUN, evaluates on the test split, logs GPU energy |
| `energy.py` | Turns the GPU power log into kWh and EV-km, appends to `experiments.csv` |

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
