"""Evaluate checkpoints on Pothole-Mix test (overall + per subset) and zero-shot on all 50 NRD images.

    python3 -m segmentation.evaluate /cluster/work/$USER/tdt17/runs/<run>/best.pt [more best.pt ...]

Writes <run>/eval.json and <run>/eval_figs/, and appends one row per checkpoint to
results/segmentation/eval_results.csv (commit that file; it is the results table for the slides).
Images are evaluated at the training scale (long side capped at the config's max_long_side).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from segmentation.dataset import NorwegianRoadDamage, PotholeMix
from segmentation.train import DEVICE, REPO, Transform, evaluate, predict

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])  # A.Normalize defaults
REGION_MIN_COVER = 0.05   # an NRD region counts as found if >= 5% of its pixels are predicted (labels are loose polygons)
IMAGE_MIN_FRACTION = 0.001  # an image is "predicted damaged" if >= 0.1% of its pixels are predicted


def region_hits(gt, pred, min_cover=REGION_MIN_COVER):
    """(#ground-truth regions, #regions with >= min_cover of their pixels predicted)."""
    labels, n = ndimage.label(gt)
    if n == 0:
        return 0, 0
    cover = np.bincount(labels[pred].ravel(), minlength=n + 1)[1:] / np.bincount(labels.ravel())[1:]
    return n, int((cover >= min_cover).sum())


def append_row(table, row):
    """Append to a results CSV; rewrites the header if new columns appear (old rows get blanks)."""
    rows = list(csv.DictReader(open(table))) if table.exists() else []
    cols = list(dict.fromkeys([*(rows[0].keys() if rows else []), *row]))
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows + [row])


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None, classes=1)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt["epoch"]


@torch.no_grad()
def nrd_predict(model, ds, device, indices=None):
    """[(x, gt, pred)] for the given NRD indices, plus per-image inference times (s, first call skipped)."""
    records, times = [], []
    for k, i in enumerate(range(len(ds)) if indices is None else indices):
        x, y = ds[i]
        t0 = time.perf_counter()
        p = (torch.sigmoid(predict(model, x[None].to(device)))[0, 0] > 0.5).cpu().numpy()  # .cpu() syncs the GPU
        if k:
            times.append(time.perf_counter() - t0)
        records.append((x, y.numpy() > 0, p))
    return records, times


def nrd_metrics(records, times, fig_path, title="NRD zero-shot"):
    """Micro pixel metrics + region recall + image-level accuracy over all records, and an example figure."""
    tp = fp = fn = regions = hits = 0
    img = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_image, area_gt, area_pred = [], [], []
    for _, g, p in records:
        tp += int((p & g).sum()); fp += int((p & ~g).sum()); fn += int((~p & g).sum())
        if g.any():
            per_image.append((p & g).sum() / (p | g).sum())
        area_gt.append(100 * g.mean()); area_pred.append(100 * p.mean())  # damage area, % of image
        n, h = region_hits(g, p)
        regions += n; hits += h
        pred_pos, gt_pos = p.mean() >= IMAGE_MIN_FRACTION, g.any()
        img[{(1, 1): "tp", (1, 0): "fp", (0, 1): "fn", (0, 0): "tn"}[(int(pred_pos), int(gt_pos))]] += 1

    eps = 1e-9
    metrics = {
        "iou": tp / (tp + fp + fn + eps), "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps), "recall": tp / (tp + fn + eps),
        "region_recall": hits / max(regions, 1), "regions": regions,
        "image_accuracy": (img["tp"] + img["tn"]) / len(records), "image_confusion": img,
        "miou_img": float(np.mean(per_image)) if per_image else float("nan"),
        "area_pct_gt": float(np.mean(area_gt)), "area_pct_pred": float(np.mean(area_pred)),
        "area_pct_mae": float(np.mean(np.abs(np.array(area_pred) - np.array(area_gt)))),
        "ms_per_image": 1000 * float(np.mean(times)) if times else None,
    }

    # Qualitative figure: the 6 images with most labelled damage + 2 without any.
    shown = sorted(records, key=lambda r: -r[1].mean())
    pick = shown[:6] + [r for r in shown if not r[1].any()][:2]
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    for ax, (x, g, p) in zip(axes.flat, pick):
        rgb = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        rgb[p] = 0.5 * rgb[p] + 0.5 * np.array([1, 0, 0])
        ax.imshow(rgb); ax.contour(g, levels=[0.5], colors="lime", linewidths=1); ax.axis("off")
    fig.suptitle(f"{title}: prediction in red, label outline in green (last two: no labelled damage)")
    plt.tight_layout(); plt.savefig(fig_path, dpi=110); plt.close(fig)
    return metrics


def eval_nrd(model, ds, device, fig_path):
    return nrd_metrics(*nrd_predict(model, ds, device), fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    device = DEVICE

    for ckpt_path in map(Path, args.checkpoints):
        model, cfg, epoch = load(ckpt_path, device)
        run = ckpt_path.parent
        (run / "eval_figs").mkdir(exist_ok=True)
        tf = Transform(cfg["max_long_side"])

        test = PotholeMix("testing", transform=tf)
        loader = lambda idx: DataLoader(Subset(test, idx), 1, num_workers=args.num_workers)
        pm = {"all": evaluate(model, loader(list(range(len(test)))), device)}
        for sub in sorted({s for *_, s in test.items}):
            pm[sub] = evaluate(model, loader([i for i, (*_, s) in enumerate(test.items) if s == sub]), device)

        nrd = eval_nrd(model, NorwegianRoadDamage(transform=tf), device, run / "eval_figs/nrd_examples.png")

        result = {"checkpoint": str(ckpt_path), "epoch": epoch, "device": str(device),
                  "pothole_mix_test": pm, "nrd_zero_shot": nrd}
        (run / "eval.json").write_text(json.dumps(result, indent=2))

        row = {"run": run.name, "arch": cfg["arch"], "encoder": cfg["encoder"], "best_epoch": epoch,
               **{f"pm_test_{k}": round(v, 4) for k, v in pm["all"].items()},
               **{f"nrd_{k}": round(nrd[k], 4) for k in ("iou", "dice", "precision", "recall", "region_recall", "image_accuracy",
                                                    "miou_img", "area_pct_gt", "area_pct_pred", "area_pct_mae")},
               "nrd_ms_per_image": round(nrd["ms_per_image"], 1) if nrd["ms_per_image"] else None,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type}
        append_row(REPO / "results/segmentation/eval_results.csv", row)

        print(f"\n== {run.name} (epoch {epoch})")
        for name, m in pm.items():
            print(f"  PM test {name:28s} IoU {m['iou']:.3f}  Dice {m['dice']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  mIoU/img {m['miou_img']:.3f}")
        print(f"  NRD zero-shot  IoU {nrd['iou']:.3f}  Dice {nrd['dice']:.3f}  P {nrd['precision']:.3f}  R {nrd['recall']:.3f}  "
              f"region recall {nrd['region_recall']:.3f} ({nrd['regions']} regions)  image acc {nrd['image_accuracy']:.3f}  "
              f"mIoU/img {nrd['miou_img']:.3f}  area% pred {nrd['area_pct_pred']:.2f} vs gt {nrd['area_pct_gt']:.2f}  "
              f"{nrd['ms_per_image']:.1f} ms/img")


if __name__ == "__main__":
    main()
