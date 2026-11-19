"""Binary road-damage segmentation: train on Pothole-Mix, select on its validation split.

    python3 -m segmentation.train --config segmentation/configs/baseline.yaml
    python3 -m segmentation.train --config segmentation/configs/baseline.yaml --set limit=16 epochs=1 num_workers=0   # smoke test
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from common.paths import DATA_ROOT
from segmentation.dataset import PotholeMix, RDD2022Boxes
from segmentation.losses import box_projection_loss

REPO = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Transform:
    """Downscale-only to max_long_side (keeps thin cracks visible, never upsamples small images), then albumentations."""

    def __init__(self, max_long_side, crop=None, binary=True):
        self.max_long_side = max_long_side
        self.binary = binary  # False keeps mask values (box ids for the projection loss); resizing is nearest anyway
        aug = []
        if crop:  # training
            aug += [
                A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_CONSTANT),
                # Most crops contain damage, some don't, so the model also learns "no damage here".
                A.OneOf([A.CropNonEmptyMaskIfExists(crop, crop, p=0.7), A.RandomCrop(crop, crop, p=0.3)], p=1),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
            ]
        self.aug = A.Compose(aug + [A.Normalize(), ToTensorV2()])

    def __call__(self, image, mask):
        s = self.max_long_side / max(image.shape[:2])
        if s < 1:
            size = (round(image.shape[1] * s), round(image.shape[0] * s))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        out = self.aug(image=image, mask=(mask > 0).astype(np.uint8) if self.binary else mask)
        return {"image": out["image"], "mask": out["mask"].float()}


def predict(model, x):
    """Logits for a batch of any size: pad to a multiple of 32 (U-Net requirement), crop back."""
    h, w = x.shape[-2:]
    ph, pw = -h % 32, -w % 32
    return model(F.pad(x, (0, pw, 0, ph), mode="replicate"))[..., :h, :w]


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """Binary metrics. iou/dice/precision/recall are dataset-level (micro: TP/FP/FN summed over all pixels).
    miou_img is IoU per image averaged over images that contain damage (the "foreground mIoU" used by
    Shami et al. 2026, so our numbers can be put next to theirs)."""
    model.eval()
    tp = fp = fn = 0
    per_image = []
    for x, y in loader:
        x, y = x.to(device), y.to(device).bool()
        p = torch.sigmoid(predict(model, x)).squeeze(1) > threshold
        tp += (p & y).sum().item()
        fp += (p & ~y).sum().item()
        fn += (~p & y).sum().item()
        has = y.flatten(1).any(1)
        inter, union = (p & y).flatten(1).sum(1), (p | y).flatten(1).sum(1)
        per_image += (inter[has] / union[has]).tolist()
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "miou_img": float(np.mean(per_image)) if per_image else float("nan"),
    }


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides:
        k, v = kv.split("=", 1)
        if k not in cfg and k != "limit":
            raise KeyError(f"unknown config key: {k}")
        v = yaml.safe_load(v)
        if isinstance(cfg.get(k), float) and isinstance(v, (int, str)):
            v = float(v)  # YAML reads "1e-4" (no dot) as a string
        cfg[k] = v
    return cfg


def limited(ds, n):
    # Evenly spaced, so every Pothole-Mix subset is represented in a smoke test.
    return Subset(ds, np.linspace(0, len(ds) - 1, n).astype(int).tolist()) if n else ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. epochs=1 lr=1e-4")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = DEVICE
    use_amp = device.type == "cuda"

    out = Path(os.environ.get("OUT_ROOT", REPO / "runs")) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"device={device} data={DATA_ROOT} out={out}")

    projection = cfg.get("box_loss") == "projection"  # only meaningful for box-labelled data (rdd_norway)
    train_tf = Transform(cfg["max_long_side"], cfg["crop"], binary=not projection)
    val_tf = Transform(cfg["max_long_side"])
    dataset = cfg.get("dataset", "pothole_mix")
    if dataset == "pothole_mix":
        train_ds, val_ds = PotholeMix("training", transform=train_tf), PotholeMix("validation", transform=val_tf)
    elif dataset == "rdd_norway":  # weak labels: filled detection boxes
        train_ds = RDD2022Boxes("Norway", "train", transform=train_tf, instance_ids=projection)
        val_ds = RDD2022Boxes("Norway", "val", transform=val_tf)  # val metrics stay against filled boxes
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if projection and dataset != "rdd_norway":
        raise ValueError("box_loss: projection needs box labels (dataset: rdd_norway)")
    train_ds, val_ds = limited(train_ds, cfg.get("limit")), limited(val_ds, cfg.get("limit"))
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                          drop_last=len(train_ds) > cfg["batch_size"], pin_memory=use_amp,
                          persistent_workers=cfg["num_workers"] > 0)
    val_dl = DataLoader(val_ds, 1, num_workers=cfg["num_workers"])  # batch 1: validation images differ in size

    init = cfg.get("init_checkpoint")  # continue from an earlier run (e.g. Pothole-Mix -> RDD-Norway)
    model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], classes=1,
                             encoder_weights=None if init else cfg["encoder_weights"])
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
    model.to(device)
    dice = smp.losses.DiceLoss("binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, t_start = -1.0, 0, time.time()
    with open(out / "metrics.csv", "w", newline="") as f:
        log = csv.DictWriter(f, ["epoch", "train_loss", "val_iou", "val_dice", "val_precision", "val_recall", "val_miou_img", "lr", "seconds"])
        log.writeheader()
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            model.train()
            losses = []
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    if projection:
                        loss = box_projection_loss(logits, y)
                    else:
                        loss = dice(logits, y.unsqueeze(1)) + bce(logits, y.unsqueeze(1))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()

            val = evaluate(model, val_dl, device)
            row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                   **{f"val_{k}": round(v, 4) for k, v in val.items()},
                   "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
            log.writerow(row); f.flush()
            print(row, flush=True)
            # Projection runs keep the last epoch: val IoU against *filled* boxes would reward filling the boxes,
            # which is exactly what the projection loss avoids.
            if val["iou"] > best or projection:
                best, best_epoch = val["iou"], epoch
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "best.pt")
            elif cfg.get("patience") and epoch - best_epoch >= cfg["patience"]:
                print(f"early stop: no val IoU improvement for {cfg['patience']} epochs (best epoch {best_epoch})")
                break

    summary = {  # for the sustainability section: GPU-hours x power = energy
        "best_val_iou": best,
        "checkpoint_selection": "last epoch" if projection else "best val IoU",
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_seconds": round(time.time() - t_start, 1),
        "device": torch.cuda.get_device_name() if use_amp else device.type,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "power_log": os.environ.get("POWER_LOG"),  # set by segmentation/train_gpu.sh (non-slurm runs)
        "data_root": str(DATA_ROOT),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
