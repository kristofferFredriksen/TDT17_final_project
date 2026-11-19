# TDT17 mini-project: road damage detection and segmentation

NTNU TDT17 Visual Intelligence, Task 4, option 1 "Road Infrastructure". Group of two.

| Task | Folder | Data |
|---|---|---|
| A. Object detection (D00, D10, D20, D40) | [`detection/`](detection/README.md) | RDD2022, evaluated on Norwegian Road Damage |
| B. Semantic segmentation (damage vs background) | [`segmentation/`](segmentation/README.md) | Pothole-Mix + RDD2022 Norway, evaluated on Norwegian Road Damage |

```
common/paths.py      dataset paths shared by both tasks
detection/           code, slurm jobs, requirements, results for task A
segmentation/        code, configs, slurm jobs, requirements, results for task B
presentation/        talk scripts and the figures they show
```

Never committed: data (`/data/`), checkpoints (`*.pt`), `runs/`, `logs/`, `slurm-*.out`.

## Data

All datasets are on IDUN under `/cluster/projects/vc/courses/TDT17/2026/1_infra/`. `common/paths.py` looks for them
in `$DATA_ROOT`, then in `<repo>/data/` (same structure as on IDUN), then in the IDUN course folder.

Copy to a local machine (needs the NTNU network or VPN):

```bash
R=<ntnu-user>@idun.hpc.ntnu.no:/cluster/projects/vc/courses/TDT17/2026/1_infra
PM=Pothole-Mix-v2/kfth5g2xk3-2/Pothole-Mix
mkdir -p data/$PM data/RDD2022/RDD2022
rsync -avh $R/NorwegianRoadDamage data/                                  # 10 MB
rsync -avh "$R/$PM/pothole-mix-v1.0-20220526.zip*" data/$PM/               # 3.4 GB, then:
(cd data/$PM && sha256sum -c pothole-mix-v1.0-20220526.zip.sha256sum && unzip -q pothole-mix-v1.0-20220526.zip && rm pothole-mix-v1.0-20220526.zip*)
rsync -avh $R/RDD2022/RDD2022/Norway data/RDD2022/RDD2022/                 # 11 GB
```
