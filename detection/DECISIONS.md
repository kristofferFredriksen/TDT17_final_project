# Task A: Decision log

Every choice that shapes the detection results, with the reasoning and the evidence behind it.
The numbers come from `prepare_rdd.py` / `eda_rdd.py` (plots in the `eda` folder) unless stated otherwise.
Each entry: **Decision**, **Why**, **Alternatives**, and **Consequences / limitations**.

---

## Data

### D1. Keep only the four target classes (D00, D10, D20, D40)
- **Decision:** Drop every other RDD2022 label: D44 (5,057), D50 (3,581), D43 (793), Repair (277),
  D01 (179), D11 (45), D0w0 (1). In total that is 9,933 boxes, after China_Drone is excluded.
- **Why:**
  - The task specifies these four classes.
  - Most of the dropped labels are not damage: D43 and D44 are faded crosswalks and white lines,
    D50 is a manhole, D01 and D11 are construction joints, and Repair is patched road.
  - They also only appear in some countries: Japan (8,284 of the 9,933), India and China_MotorBike.
    Keeping them would teach the model country-specific classes that the customer doesn't use.
  - D0w0 is a single typo.
- **Consequences:**
  - Images that only had dropped labels become background images.
  - Faded markings and manholes are now implicitly "not damage". That is the behaviour we want,
    but it may give false positives on similar-looking structures.

### D2. Exclude China_Drone
- **Decision:** Leave out the 2,401 China_Drone images (3,068 target boxes, about 5.6% of them).
- **Why:**
  - These are top-down drone images at 512×512. Every other subset is a vehicle-mounted camera
    looking forward along the road. The customer's data (NorwegianRoadDamage) is also dashcam video.
  - China_Drone has no official test folder either.
  - Mixing the two viewpoints asks one model to learn two different appearances of the same damage.
- **Alternative:** Include it, as an ablation if time allows.

### D3. Don't use the official RDD2022 test folders
- **Decision:** Only use `train/` from each country.
- **Why:** The test labels were never released (47,420 images but only 38,385 XML files), so they
  can't be evaluated.

### D4. Our own 80/10/10 train/val/test split, stratified per country
- **Decision:**
  - Each country is shuffled and split separately, with seed 42.
  - Result: 28,786 / 3,599 / 3,599 images.
  - The split is saved in `images.csv`, so it's reproducible.
- **Why:**
  - Every country is represented in every split in the same proportion, which makes per-country
    evaluation possible.
  - Class shares come out almost identical across splits (D00 is 46.7%, 47.5% and 46.9% of boxes
    in test, train and val), so test results are not skewed by a chance difference in class mix.
- **Use of each split:**
  - **Val** selects `best.pt` and drives early stopping.
  - **Test** is only used for the final reported numbers.
- **Limitation:** The split is random at image level. RDD images are often taken in sequences
  along the same drive, so near-identical neighbouring images can end up in both train and test,
  which may inflate the test scores. RDD2022 has no sequence IDs, so a group-wise split isn't
  possible. Mention this in the discussion.

### D5. Keep the background images (no damage)
- **Decision:** Keep all images without target boxes, about 14,100 of 35,984 (39%).
- **Why:**
  - Real inspection footage is mostly undamaged road. Background images teach the model not to fire
    on shadows, markings and patches, which reduces false positives.
  - The share varies a lot between countries: Norway 64%, Czech 62%, India 58%, Japan 25%,
    China_MotorBike 2%, US 0%. That shows the subsets were curated differently.
- **Alternative:** Subsample the background images. This is a possible later experiment.

### D6. Convert VOC to YOLO format, symlink the images, and parse the labels once
- **Decision:** `prepare_rdd.py` does the following in a single pass:
  - Writes YOLO `.txt` labels (class cx cy w h, normalised).
  - Symlinks the images instead of copying them.
  - Caches all annotations in `boxes.csv` and `images.csv`.
- **Why:**
  - Ultralytics only reads YOLO format.
  - Symlinks avoid duplicating 14 GB.
  - IDUN's network filesystem is slow on many small files: a first single-threaded pass over the
    XML files took several minutes. Parsing once with 32 threads and caching to CSV makes the data
    analysis run in seconds.

### D7. NorwegianRoadDamage is a held-out evaluation set only
- **Decision:** Never train on it. Evaluate on all 50 images (its train and test folders together)
  in two ways:
  - **`nrd_agnostic`:** all 109 boxes count as "damage", and predictions match regardless of class.
    This is the main customer metric.
  - **`nrd_perclass`:** only the 17 D00, 4 D20 and 4 D40 boxes.
- **Why:**
  - It is the dataset that matters most to the customer, but there are only 50 images. That is too
    few to train on, and enough for an indicative test.
  - 84 of its 109 annotations have only a generic class (category ID 4,
    "Norwegian-Road-Damage-Masks"), and there is no D10 at all. Per-class mAP over the four classes
    is therefore not possible.
  - Its own train and test folders share frames from 3 of the 6 source videos, so they are not
    independent. Using all 50 images for evaluation only avoids this issue.
- **Consequences:**
  - **`nrd_perclass`:** predictions on the generic regions count as false positives, so recall is
    meaningful while precision and mAP are underestimates.
  - **Domain shift:** the NRD frames are 1280×720 video, a different camera from the RDD2022 Norway
    images (about 4040×2035). Performance on NRD measures cross-dataset generalisation.
- **Visual check (`visualize_nrd.py`, baseline `yolo11s_640`):**
  - **Category 4 is a mix,** not one hidden class. About a third of the boxes are wide strips across
    the lane (36% have w/h > 2), which look like transverse cracks. The rest are patches and broken
    edges along the shoulder, plus small isolated spots. It can't be mapped reliably to D10, which
    confirms that class-agnostic evaluation is the right main metric.
  - **NRD's D00 boxes follow a different convention from RDD's.** In RDD, D00 boxes are tall. In NRD,
    0 of 17 are tall and 29% are wide, because annotators box a whole diagonal crack running along the
    lane. A model that partly uses box shape to tell D00 from D10 will be confused by this.
  - **NRD's D40 boxes are tiny:** a median shortest side of 9 px at native 1280×720, so about 4–5 px
    at 640. That is too small at the baseline resolution.
  - **The baseline mostly misses damage rather than boxing it differently.**
    - At conf ≥ 0.25 it makes only 15 predictions across all 50 images.
    - Of the 109 ground-truth boxes, 90 have no overlapping prediction, 7 have IoU below 0.1,
      4 have IoU 0.1–0.5 and 8 have IoU ≥ 0.5.
    - The hits it does make have low confidence (0.27–0.64), so the model is under-confident on this
      domain.
  - **The images themselves are a different domain:** the dashboard covers the bottom ~25% of each
    frame, the light is flat overcast early spring with snow at the roadside, and the damage is subtle.

---

## Method

### M1. Baseline: Ultralytics YOLO11s, fine-tuned from COCO weights
- **Why:**
  - A well-documented, widely used one-stage detector that is quick to train.
  - The "s" size is a reasonable trade-off between accuracy and GPU time given the deadline.
  - Pretrained COCO weights give general visual features, so the model converges much faster than
    from scratch.

### M2. Experiments change one thing at a time from the baseline
| Run | Change | Question | Motivation |
|---|---|---|---|
| `yolo11s_640` | – | Reference | – |
| `yolo26s_640` | Architecture | Is the newer model better on this data? | YOLO26 (Jan 2026, Ultralytics 8.4) is NMS-free and claims better small-object detection. To be verified, not assumed. |
| `yolo11s_1280` | Input resolution | Does resolution help, especially in Norway? | The RDD Norway images are about 4040 px wide. At 640 px, half of Norway's boxes have a shortest side under 16 px, and **23.5% are under 8 px**. Other countries are at 0–0.3%. |

- The 1280 run uses batch size 16 instead of 32, to fit in GPU memory, so it differs from the
  baseline in two ways. This is acknowledged as a confounder.

### M3. Training settings: Ultralytics defaults, with three exceptions
- **`epochs=100, patience=20`:** stop if val fitness (mainly mAP50-95) has not improved in 20
  epochs. This saves GPU time and energy when the model has converged.
- **`seed=0, deterministic=True`:** for reproducibility.
- **`batch=32`** at 640 px, and 16 at 1280 px.

### M4. Augmentation: keep the defaults, after checking they suit the classes
- **Decision:** Keep `fliplr=0.5`, `flipud=0`, `degrees=0` and mosaic, with `close_mosaic=10`.
- **Why:**
  - D00 and D10 are distinguished by orientation. D00 boxes are tall (along the road) and D10 boxes
    are wide (across the road); see `aspect_ratio_per_class.png`.
  - A horizontal flip keeps that orientation.
  - A vertical flip or 90° rotation would make a D00 look like a D10 while the label still says
    D00, so those must stay off.
  - Damage only appears in the lower part of the image (`box_centers_per_class.png`). A vertical
    flip would also put road in the sky region.

### M5. Evaluation protocol (`evaluate.py`)
- **Metrics:** Precision, Recall, mAP50 and mAP50-95, as suggested in the task, plus AP50 and recall
  per class, and inference time per image.
- **Sets:**
  - The whole test split
  - **Each country separately.** The class mix differs strongly: Norway is 76% D00, India is mostly
    D40, Japan is mostly D20. An overall number hides this.
  - The two NRD variants (D7)
- **Resolution:** Each model is evaluated at the resolution it was trained at.
- **Verification:** The class-agnostic mode (`single_cls`) was checked with a pretrained COCO model.
  A correct box with the wrong class gives mAP50 ≈ 1.0 in class-agnostic mode and counts as missed
  otherwise.

---

## Infrastructure and sustainability

### S1. Energy measurement
- **Decision:** `nvidia-smi` logs GPU power every 5 s during every training and evaluation job.
  `energy.py` integrates the log into kWh and converts it to EV kilometres (0.18 kWh/km).
- **Limitation:** Only the GPU is measured: CPU, RAM and cooling are not included. `--pue` can scale
  the number for data-centre overhead.

### S2. More CPUs instead of more GPUs
- **Observation:** The baseline (A100-PCIE-40GB, 8 CPUs) ran at 71–75% GPU utilisation, with power
  swinging between 125 and 255 W, and used 10.7 of 40 GB of GPU memory. The GPU was waiting for data
  (decoding the large Norway JPEGs and building the augmented images).
- **Decision:** From the second run on, use 16 CPUs (dataloader workers = CPUs / 2, because
  validation uses twice as many workers as training). Don't use several GPUs.
- **Why not several GPUs:** They wouldn't fix a data-loading bottleneck, they use the same or more
  energy in total, and they wait longer in the queue.
- **Note:** This doesn't change the results, only the speed. Compare utilisation between runs for
  the sustainability section.
- **Supporting evidence:** Ultralytics warned "Slow image access detected (read: 4.9 MB/s)" on
  `/cluster/work`, which confirms that reading images from network storage is a bottleneck.
  A possible fix is `cache=ram` (decoded images kept in RAM, about 30 GB at 640 px, so it needs
  `--mem=64G`) or copying the data to the node's local disk.

### S3. Train only on A100s
- **Observation:** IDUN's GPUQ mixes P100, V100, A100, H100 and H200 nodes. An evaluation job landed
  on a V100 (compute capability 7.0) and crashed: our PyTorch build (CUDA 13) only supports compute
  capability 7.5 and newer.
- **Decision:**
  - Training jobs: `--constraint=a100`.
  - Evaluation jobs: `--constraint="a100|h100|h200"`.
- **Why A100 only for training:** Besides compatibility, using one GPU type makes training time and
  energy directly comparable between runs.

---

## Run log

| Run name | Slurm job | Model | imgsz | batch | CPUs | Notes |
|---|---|---|---|---|---|---|
| `yolo11s_640` | 26241819 | yolo11s.pt | 640 | 32 | 8 | Baseline, A100-PCIE-40GB, about 1:55 min per epoch. Early-stopped at epoch 96 (best: 76), 2.48 h, 0.49 kWh. Val: mAP50 0.602, mAP50-95 0.306. **Test: mAP50 0.596, mAP50-95 0.304** (D40 weakest: AP50 0.478, R 0.421) |
| `yolo26s_640` | | yolo26s.pt | 640 | 32 | 16 | |
| `yolo11s_1280` | 26241981 | yolo11s.pt | 1280 | 16 | 16 | 24 h time limit; about 5:10 min per epoch (about 3× the baseline), 16.1 GB GPU memory |

## Open questions / possible next steps
- Visually inspect NRD category 4.
- Subsample background images (D5).
- Tiling for the large Norway images. Ultralytics 8.4.215 has a `tile` option, to be investigated.
- RT-DETR as a transformer-based comparison.
- Near-duplicate images across splits (D4 limitation). China_MotorBike scores suspiciously high
  (baseline test mAP50 0.93, D40 recall 1.0, compared with 0.60 overall), which is worth checking for
  near-duplicates between train and test.
- NRD operating point: the model is under-confident on NRD, so for the customer, report recall and
  precision at a lower confidence threshold, chosen on RDD val, alongside mAP.
