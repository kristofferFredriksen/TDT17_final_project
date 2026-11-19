# Task B: Road damage segmentation

Binary segmentation (damage vs background) with U-Net and SegFormer from
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), trained on Pothole-Mix
(and RDD2022 Norway boxes as weak labels), evaluated on all 50 Norwegian Road Damage (NRD) images.

| File | What it does |
|---|---|
| `dataset.py` | `PotholeMix`, `NorwegianRoadDamage` (NRD), `RDD2022Boxes` (detection boxes as weak masks) |
| `train.py` | Training from a config yaml (`--set key=value` overrides); also `Transform`, `predict`, `evaluate` |
| `losses.py` | Box projection loss (BoxInst-style) for box-labelled data |
| `evaluate.py` | Pothole-Mix test (overall + per subset) and NRD zero-shot; appends to `results/eval_results.csv` |
| `finetune_nrd.py` | Leave-one-video-out (LOVO) fine-tuning on NRD, metrics pooled over all 50 images; appends to `results/finetune_results.csv` |
| `run_stats.py` | Per-run speed, GPU utilisation, power and energy |
| `configs/` | `baseline` (U-Net R34), `segformer_b2`, `unet_r34_long` (early stopping), `unet_r34_rdd_norway` (stage 2, filled boxes), `unet_r34_rdd_norway_proj` (stage 2, projection loss) |
| `eda.ipynb` | Data analysis; figures go to `results/eda/` |
| `train.slurm`, `run.slurm` | IDUN jobs (A100 only, power logging) |
| `train_gpu.sh` | Background training on a GPU machine without slurm, with power logging |
| `results/` | Result CSVs and figures (the only outputs that are committed) |

## Setup

Python 3.11. From the repo root:

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r segmentation/requirements.txt
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

RTX 5090 (Blackwell) needs a CUDA 12.8+ torch build; the default PyPI wheel has it. On IDUN the slurm scripts use
the conda env `tdt17` and require `--constraint=a100` (the torch build has no kernels for P100/V100).

## Running

Everything runs from the repo root as modules. Outputs go to `$OUT_ROOT` or `runs/`.

```bash
python3 -m segmentation.train --config segmentation/configs/baseline.yaml --set limit=16 epochs=1 num_workers=2  # smoke test
segmentation/train_gpu.sh 0 segmentation/configs/baseline.yaml                                 # stage 1, local GPU
segmentation/train_gpu.sh 0 segmentation/configs/unet_r34_rdd_norway_proj.yaml --set init_checkpoint=runs/<stage1>/best.pt
python3 -m segmentation.evaluate runs/<run>/best.pt [more ...]                                 # PM test + NRD zero-shot
python3 -m segmentation.finetune_nrd runs/<run>/best.pt --init pothole_mix --seed 0             # LOVO on NRD
python3 -m segmentation.finetune_nrd runs/<run>/best.pt --init imagenet --seed 0                # ablation: no Pothole-Mix
python3 -m segmentation.run_stats runs                                                         # time and energy per run
```

On IDUN: `sbatch segmentation/train.slurm` (or `CONFIG=... sbatch segmentation/train.slurm --set ...`) and
`sbatch segmentation/run.slurm segmentation.evaluate <ckpt>`.

## Data facts that matter

- **Pothole-Mix** masks are RGB: green = crack, red = pothole. crack500 images have EXIF orientation tags but the
  masks match the raw pixels, so EXIF rotation is never applied.
- **NRD**: 50 dashcam frames (1280x720) from 6 videos, loose polygons around damaged areas, mostly a generic
  "damage" class. The official train/test split leaks (every test video is also in train), so it is not used.
- **RDD2022 Norway**: 8161 labelled images (3650x2044), only boxes; 64% have no box. Train/val = first 90% / last
  10% by file name, so frames of the same road don't end up in both.

## Method decisions

- **Binary segmentation**, because NRD labels are mostly an untyped "damage" class.
- **NRD is only for evaluation**: zero-shot on all 50 images, and LOVO fine-tuning (6 folds, one video held out
  each, metrics pooled over the 50 held-out predictions).
- **Downscale only, to long side 1024**, then random 512 crops (70% containing damage): at 512 px thin cracks
  shrink to ~1.5 px. Dice + BCE loss, since damage is 0.3–3% of the pixels.
- **U-Net + ResNet-34 (24.4 M params) vs SegFormer-B2 (24.7 M)**: same size, CNN vs transformer.
- **40 epochs, cosine LR to 0, keep the best val-IoU checkpoint.** Early stopping with a longer schedule was worse
  (it stops while the LR is still high).
- **Stage 2 on RDD2022 Norway**: filled boxes overstate the damage area, so there is also a box projection loss
  (outside boxes = background, no pixel loss inside, every box row/column must contain damage). Projection runs
  keep the **last** epoch, because val is scored against filled boxes and would reward filling them.
- **Metrics**: micro IoU/Dice/precision/recall (all pixels pooled), per-image mIoU over images with damage (the
  background paper's "foreground mIoU"), region recall (a region counts as found if ≥ 5% of it is predicted),
  image accuracy (45/50 NRD images have damage, so "always damage" already scores 0.90), damage-area error, ms/image.
