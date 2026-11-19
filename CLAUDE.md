# TDT17 Mini-Project – Road damage segmentation (segmentation branch)

Context for humans and Claude Code sessions picking this up on a new machine.
Last updated 2026-11-19. **Deadline: Fri 2026-11-27 23:59** (14-min video for the group of two).

## Who does what
- Course project TDT17 (NTNU), Task 4, option 1 "Road Infrastructure". Group of two.
- **This branch (`segmentation`) = semantic segmentation** (Kristoffer). The teammate owns **object detection on
  RDD2022** on the `detection` branch – do not touch `detection/` or that branch.
- Assignment PDF and background paper live outside the repo (`../TDT17_Task4_Mini_Project_2026_v1.pdf`,
  `../Mamoona_RDD-Seg_2026.pdf`).

## Conventions
- Always write run commands as **`python3`**, run from the repo root as modules: `python3 -m segmentation.train ...`.
- Short conventional commits (`feat(seg): ...`, `fix(slurm): ...`, `results(seg): ...`).
- Never commit data, checkpoints or logs (`/data/`, `runs/`, `logs/`, `*.pt`, `slurm-*.out` are gitignored).
  Commit small results only: `results/segmentation/*.csv` and figures.
- Each new script gets a small assert-based test (`segmentation/test_*.py`, run as `python3 -m segmentation.test_x`).

## Repo layout
```
common/paths.py              dataset paths: $DATA_ROOT > <repo>/data (if it exists) > IDUN course dir
segmentation/
  dataset.py                 PotholeMix, NorwegianRoadDamage (NRD), RDD2022Boxes (detection boxes as weak masks)
  train.py                   training (config yaml + --set key=value overrides), Transform, predict, evaluate
  losses.py                  box projection loss (BoxInst-style) for box-labelled data
  evaluate.py                Pothole-Mix test (overall + per subset) + NRD zero-shot; appends results CSV
  finetune_nrd.py            leave-one-video-out (LOVO) fine-tuning on NRD, pooled over all 50 images
  preshrink.py               copy of Pothole-Mix capped at 1024 px (speed/storage; same model input)
  run_stats.py               per-run speed, GPU util, power, energy (from power_<jobid>.csv)
  configs/*.yaml             baseline (U-Net R34), segformer_b2, unet_r34_long (early stop),
                             unet_r34_rdd_norway (stage 2, filled boxes), unet_r34_rdd_norway_proj (stage 2, projection loss)
  test_*.py                  tests (synthetic data, no dataset needed)
eda/seg_eda.ipynb            EDA notebook (figures -> results/segmentation/eda/)
slurm/seg_train.slurm        IDUN training job (A100 only, power logging)
slurm/seg_run.slurm          IDUN generic job: `sbatch slurm/seg_run.slurm <module> <args>`
scripts/train_gpu.sh         background training on a non-slurm GPU box (vc04 style)
results/segmentation/        eval_results.csv, finetune_results.csv (+ *_v1.csv older metric set), figs/, eda/
```

## Data
| Dataset | Use | Where on IDUN (`/cluster/projects/vc/courses/TDT17/2026/1_infra/`) | Size |
|---|---|---|---|
| Pothole-Mix | stage-1 training; per-subset test | `Pothole-Mix-v2/kfth5g2xk3-2/Pothole-Mix/pothole-mix/{training,validation,testing}/<subset>/{images,masks}` | 3.7 GB (zip 3.4 GB in same folder) |
| Norwegian Road Damage (NRD) | **evaluation target** (+ LOVO fine-tuning) | `NorwegianRoadDamage/{train,test}/_annotations.coco.json` | 10 MB |
| RDD2022 Norway | stage-2 training (boxes -> weak masks) | `RDD2022/RDD2022/Norway/train/{images,annotations/xmls}` | several GB (8161 images, 3650x2044) |

**Dataset facts that matter:**
- Pothole-Mix masks are RGB: green (0,255,0) = crack, red (255,0,0) = pothole. Pairs match by filename stem.
  crack500 has EXIF orientation tags but masks match the **raw** pixels -> never `exif_transpose`; if using cv2 use
  `IMREAD_IGNORE_ORIENTATION`. 6 subsets; cracks-and-potholes-in-road is 60% of training.
- NRD: 50 dashcam frames (1280x720) from **6 videos**. COCO polygons; category 4 is a generic "damage" class (most
  labels), no D10, 4 potholes. **The official train/test split leaks** (all test videos also in train) -> we never
  use it. Labels are **loose polygons around damaged areas** (made with SAM-3 pseudo-labels + manual correction,
  see Shami et al. 2026 = the background paper; NRD = their "NRDD").
- RDD2022 Norway: 7345 train / 816 val (val = last 10% by filename, not random). Classes only D00 8570, D10 1730,
  D20 468, D40 461 boxes; 5247 images (64%) have no box. Test folder has no labels.

### Getting the data onto a new machine (e.g. the school PC with the RTX 5090)
Put it in `<repo>/data/` with the same structure as on IDUN – `common/paths.py` then finds it automatically.
Needs NTNU network or VPN. Linux / WSL (on plain Windows use WSL, or `scp -r` instead of rsync):
```bash
cd <repo>
R=kristotf@idun.hpc.ntnu.no:/cluster/projects/vc/courses/TDT17/2026/1_infra
PM=Pothole-Mix-v2/kfth5g2xk3-2/Pothole-Mix
mkdir -p data/$PM data/RDD2022/RDD2022
rsync -avh $R/NorwegianRoadDamage data/
rsync -avh "$R/$PM/pothole-mix-v1.0-20220526.zip*" data/$PM/          # 3.4 GB, then:
(cd data/$PM && shasum -a 256 -c pothole-mix-v1.0-20220526.zip.sha256sum && unzip -q pothole-mix-v1.0-20220526.zip && rm pothole-mix-v1.0-20220526.zip*)
rsync -avh $R/RDD2022/RDD2022/Norway data/RDD2022/RDD2022/              # only needed for stage-2 runs
python3 -m segmentation.dataset                                          # sanity check: counts per subset, NRD stats
```
Trained checkpoints live only on IDUN (`/cluster/work/kristotf/tdt17/runs/<run>/best.pt`, ~100 MB each). To continue
evaluating/fine-tuning locally, copy the run folders without the figures:
```bash
rsync -avhm --include='*/' --include='best.pt' --include='*.yaml' --include='*.json' --include='*.csv' --exclude='*' \
  kristotf@idun.hpc.ntnu.no:/cluster/work/kristotf/tdt17/runs/ runs/
```

## Environment
- Python 3.11, `requirements.txt` (torch >= 2.4, smp 0.6.0, albumentations 2.0.8, ...). Tested with torch 2.15.
- New machine: `uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt`
  (or a conda env). Check: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"`.
  RTX 5090 (Blackwell, sm_120) needs a CUDA 12.8+ torch build – the default PyPI wheel of recent torch has it.
- Mac: the Intel/Rosetta Anaconda gives torch 2.2 (breaks) -> use the native `.venv` via uv.
- IDUN: conda env `tdt17` (`/cluster/work/kristotf/envs/tdt17`). Slurm scripts require `--constraint=a100` because the
  torch build has no P100/V100 kernels (compute capability < 7.5). `conda activate` must come before `set -u`.

## How to run (local GPU)
```bash
python3 -m segmentation.test_dataset && python3 -m segmentation.test_train && python3 -m segmentation.test_evaluate && python3 -m segmentation.test_losses
python3 -m segmentation.train --config segmentation/configs/baseline.yaml --set limit=16 epochs=1 num_workers=2   # smoke test
python3 -m segmentation.train --config segmentation/configs/baseline.yaml                                       # stage 1 (~22 min on A100)
python3 -m segmentation.train --config segmentation/configs/unet_r34_rdd_norway_proj.yaml --set init_checkpoint=runs/<stage1>/best.pt
python3 -m segmentation.evaluate runs/<run>/best.pt [more ...]           # PM test + NRD zero-shot -> results CSV + eval_figs/
python3 -m segmentation.finetune_nrd runs/<run>/best.pt --init pothole_mix --seed 0   # LOVO on NRD (~7 min on A100)
python3 -m segmentation.finetune_nrd runs/<run>/best.pt --init imagenet --seed 0      # ablation: no Pothole-Mix
```
Outputs go to `$OUT_ROOT` or `<repo>/runs/`. For energy numbers on a non-slurm machine, log power alongside:
`nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv -l 30 > logs/power_<name>.csv &`.
On IDUN: `sbatch slurm/seg_train.slurm` / `CONFIG=... sbatch --time=01:00:00 slurm/seg_train.slurm --set ...`,
`sbatch slurm/seg_run.slurm segmentation.evaluate <ckpt>`; results/runs on `/cluster/work/$USER/tdt17/runs`.

## Method decisions (and why)
- **Binary segmentation** (damage vs background): NRD labels are mostly an untyped "damage" class.
- **Evaluate on all 50 NRD images**: zero-shot (train on other data only) and LOVO fine-tuning (6 folds, one video
  held out each, pooled metrics over 50 images). Never train on NRD otherwise.
- Resize **downscale-only to long side 1024** + random 512 crops (70% crops containing damage): at 512 px thin cracks
  shrink to ~1.5 px (EDA). Dice + BCE loss (damage is 0.3-3% of pixels).
- U-Net + ImageNet ResNet34 (24.4 M params) vs SegFormer-B2 (24.7 M): same size, CNN vs transformer.
- 40 epochs, cosine LR to 0, keep best val-IoU checkpoint. Early stopping with a longer cosine schedule was **worse**
  (stops at high LR) -> not used.
- Metrics: micro IoU/Dice/P/R (all pixels pooled), `miou_img` (per-image IoU averaged over images with damage =
  the paper's "foreground mIoU"), NRD region recall (region found if >= 5% of it is predicted), image accuracy
  (careful: 45/50 NRD images have damage, so "always damage" = 0.90), damage-area % error, ms/image.
- Stage 2 on RDD2022-Norway: filled boxes overstate area (~7x per the paper) -> also a **box projection loss**
  (outside boxes = background; inside no pixel loss; each box row/column must contain damage). Projection runs keep
  the **last** epoch (val is against filled boxes and would reward filling).

## Results so far (A100)
Pothole-Mix test (micro IoU): U-Net 0.547, SegFormer 0.568, U-Net long/early-stop 0.523.
Val-vs-test ranking differs because the subset mix differs; U-Net is better on the dashcam-like subset.

NRD, 50 images:
| Setup | IoU | Precision | Recall | Region recall |
|---|---|---|---|---|
| Zero-shot U-Net (Pothole-Mix only) | 0.003 | 0.003 | 0.055 | 0.15 |
| LOVO U-Net, Pothole-Mix init (3 seeds) | **0.327 ± 0.007** | 0.589 | 0.424 | 0.667 |
| LOVO U-Net, ImageNet init (3 seeds) | 0.314 ± 0.025 | 0.467 | 0.493 | 0.722 |
| LOVO SegFormer, Pothole-Mix init (3 seeds) | 0.320 ± 0.010 | 0.539 | 0.441 | 0.611 |

- Zero-shot fails: false positives on sky, trees, snow, walls, bonnet; Norwegian damage (wheel-track wear, ravelling,
  patches) is not in Pothole-Mix. ~40 NRD frames fix the off-road false positives.
- Pothole-Mix init vs ImageNet init: same IoU within seed noise, but a consistent precision/recall trade-off.
- SegFormer ≈ U-Net on NRD but 25-60% slower fine-tuning and more energy -> recommend U-Net.
- Efficiency: U-Net 12 ms/img (A100), CPU ~0.7-1.0 s/img (M2); paper's SAM-3 pipeline 473 M params, 457 ms/img.
- Energy (GPU only): U-Net stage 1 0.071 kWh, SegFormer 0.119 kWh, long run 0.085 kWh. GPU util 77-83%: data
  loading was not the bottleneck (pre-shrink + 2x CPUs gave only 7%).

## Open work (as of 2026-11-19)
1. Results of the v2-metric jobs on IDUN (evaluate for baseline/SegFormer/long/RDD-filled + LOVO 3 seeds for the
   U-Net baseline and the RDD-filled model) -> `results/segmentation/*_results.csv`.
2. Projection-loss stage 2 (`unet_r34_rdd_norway_proj.yaml`) + evaluate + LOVO 3 seeds; compare `area_pct_pred`,
   precision and LOVO IoU against the filled-box run and against 0.327.
3. **Stop experimenting Sat 2026-11-21/22**; then slides/video following the assignment's 8-part structure +
   sustainability (energy table, EV-km comparison, efficiency vs SAM-3). Coordinate shared parts with the teammate.
