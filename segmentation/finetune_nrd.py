"""Leave-one-video-out (LOVO) fine-tuning on Norwegian Road Damage.

For each of the 6 NRD videos: start from a trained checkpoint, fine-tune on the frames of the other 5 videos,
predict the held-out video. Every image is predicted by a model that never saw its video, so the pooled metrics
over all 50 images are directly comparable with the zero-shot numbers from evaluate.py.

    python3 -m segmentation.finetune_nrd /cluster/work/$USER/tdt17/runs/<run>/best.pt
    python3 -m segmentation.finetune_nrd <run>/best.pt --init imagenet   # ablation: same model, no Pothole-Mix pre-training

No validation split: with 2-16 frames per video there is nothing to spare, so every fold uses the same fixed
schedule (--steps) and the final weights.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Subset

from segmentation.dataset import NorwegianRoadDamage
from segmentation.evaluate import append_row, nrd_metrics, nrd_predict
from segmentation.train import REPO, Transform


def video_of(path):
    return path.name.split("_mp4")[0]  # frames are named <video timestamp>_mp4-<frame>_jpg.rf.<hash>.jpg


def finetune(model, ds, args, device):
    use_amp = device.type == "cuda"
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    step, losses = 0, []
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= args.steps:
                break
    model.eval()
    return float(np.mean(losses[-20:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="best.pt from segmentation.train (gives arch, encoder and weights)")
    ap.add_argument("--init", choices=["pothole_mix", "imagenet"], default="pothole_mix",
                    help="pothole_mix: start from the checkpoint's weights; imagenet: same architecture, ImageNet encoder only")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=None, help="only for smoke tests")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    run = args.checkpoint.parent
    out = Path(os.environ.get("OUT_ROOT", REPO / "runs")) / f"lovo_{run.name}_{args.init}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    print(f"device={device} out={out}")

    train_ds = NorwegianRoadDamage(transform=Transform(cfg["max_long_side"], cfg["crop"]))
    eval_ds = NorwegianRoadDamage(transform=Transform(cfg["max_long_side"]))
    videos = sorted({video_of(p) for p, *_ in eval_ds.items})[:args.max_folds]

    records, times, folds, t_start = [], [], [], time.time()
    for v in videos:
        test_idx = [i for i, (p, *_) in enumerate(eval_ds.items) if video_of(p) == v]
        train_idx = [i for i, (p, *_) in enumerate(eval_ds.items) if video_of(p) != v]
        torch.manual_seed(args.seed)
        model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], classes=1,
                                 encoder_weights="imagenet" if args.init == "imagenet" else None)
        if args.init == "pothole_mix":
            model.load_state_dict(ckpt["model"])
        model.to(device)
        t0 = time.time()
        loss = finetune(model, Subset(train_ds, train_idx), args, device)
        recs, ts = nrd_predict(model, eval_ds, device, test_idx)
        m = nrd_metrics(recs, ts, out / f"fold_{v}.png", title=f"held-out video {v}")
        folds.append({"video": v, "train_frames": len(train_idx), "test_frames": len(test_idx),
                      "final_loss": loss, "seconds": round(time.time() - t0, 1), **m})
        records += recs; times += ts
        print(f"  fold {v}: {len(test_idx):2d} frames  IoU {m['iou']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"region recall {m['region_recall']:.3f}  ({folds[-1]['seconds']} s)", flush=True)

    pooled = nrd_metrics(records, times, out / "nrd_examples.png", title=f"NRD LOVO fine-tuned ({args.init} init)")
    result = {"checkpoint": str(args.checkpoint), "args": {k: str(v) for k, v in vars(args).items()},
              "pooled": pooled, "folds": folds, "seconds": round(time.time() - t_start, 1),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out / "lovo.json").write_text(json.dumps(result, indent=2))

    row = {"run": out.name, "base": run.name, "arch": cfg["arch"], "encoder": cfg["encoder"], "init": args.init, "seed": args.seed,
           "steps": args.steps, "lr": args.lr, "folds": len(folds),
           **{f"nrd_{k}": round(pooled[k], 4) for k in ("iou", "dice", "precision", "recall", "region_recall", "image_accuracy",
                                                "miou_img", "area_pct_gt", "area_pct_pred", "area_pct_mae")},
           "seconds": result["seconds"]}
    append_row(REPO / "results/segmentation/finetune_results.csv", row)

    print(f"\n== LOVO pooled over {len(records)} images ({args.init} init): IoU {pooled['iou']:.3f}  Dice {pooled['dice']:.3f}  "
          f"P {pooled['precision']:.3f}  R {pooled['recall']:.3f}  region recall {pooled['region_recall']:.3f}  "
          f"image acc {pooled['image_accuracy']:.3f}  mIoU/img {pooled['miou_img']:.3f}  "
          f"area% pred {pooled['area_pct_pred']:.2f} vs gt {pooled['area_pct_gt']:.2f}  total {result['seconds']} s")


if __name__ == "__main__":
    main()
