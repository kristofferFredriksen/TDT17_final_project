"""Visual check of a detector on NorwegianRoadDamage: ground truth vs predictions.

For each of the 50 images, draws the ground-truth polygons and boxes (coloured by category,
generic "damage" in grey) and the model's predictions (dashed, with class and confidence).
Also measures, for every ground-truth box, the best IoU with any prediction (class ignored),
which separates "missed completely" (IoU ~ 0) from "found but boxed differently" (0 < IoU < 0.5).

Runs fine on a laptop CPU (50 images).

Usage:
    python detection/visualize_nrd.py --weights results/yolo11s_640/weights/best.pt \
        --nrd-root data --out results/nrd_visual/yolo11s_640
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

CLASSES = ["D00", "D10", "D20", "D40"]
COLORS = {"D00": "#2a78d6", "D10": "#eb6834", "D20": "#1baf7a", "D40": "#eda100", "damage": "#9a9a9a"}


def load_nrd(nrd_root):
    rows = []
    for split in ("train", "test"):
        coco = json.load(open(Path(nrd_root) / split / "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            gts = []
            for a in anns.get(im["id"], []):
                name = cats[a["category_id"]]
                label = next((c for c in CLASSES if name.startswith(c)), "damage")
                x, y, w, h = a["bbox"]
                gts.append({"label": label, "box": (x, y, x + w, y + h), "seg": a["segmentation"]})
            rows.append({"split": split, "file": im["file_name"],
                         "path": Path(nrd_root) / split / im["file_name"], "gts": gts})
    return rows


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw(ax, img, gts, preds, title):
    ax.imshow(img)
    for g in gts:
        c = COLORS[g["label"]]
        for poly in g["seg"]:
            pts = np.array(poly).reshape(-1, 2)
            ax.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.35, linewidth=0)
        x1, y1, x2, y2 = g["box"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=1.5))
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        c = COLORS[p["label"]]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c,
                                   linewidth=1.5, linestyle="--"))
        ax.text(x1, y1 - 4, f"{p['label']} {p['conf']:.2f}", color="white", fontsize=6,
                bbox={"facecolor": c, "edgecolor": "none", "pad": 1})
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--nrd-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for drawn predictions")
    p.add_argument("--per-page", type=int, default=6)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 640)
    rows = load_nrd(args.nrd_root)

    matches = []
    for r in rows:
        res = model.predict(str(r["path"]), imgsz=imgsz, conf=args.conf, verbose=False)[0]
        r["preds"] = [{"label": CLASSES[int(c)], "conf": float(s), "box": tuple(b)}
                      for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
        for g in r["gts"]:
            best = max([(iou(g["box"], q["box"]), q) for q in r["preds"]], key=lambda t: t[0], default=(0.0, None))
            w, h = g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]
            matches.append({"file": r["file"], "gt_label": g["label"], "gt_w": round(w), "gt_h": round(h),
                            "best_iou": round(best[0], 3),
                            "pred_label": best[1]["label"] if best[1] else "", "pred_conf": round(best[1]["conf"], 3) if best[1] else ""})

    # Contact sheets, images with most ground truth first.
    rows.sort(key=lambda r: -len(r["gts"]))
    for page in range(0, len(rows), args.per_page):
        chunk = rows[page:page + args.per_page]
        fig, axes = plt.subplots((len(chunk) + 1) // 2, 2, figsize=(14, 4 * ((len(chunk) + 1) // 2)))
        for ax, r in zip(np.atleast_1d(axes).ravel(), chunk):
            with Image.open(r["path"]) as im:
                draw(ax, im.copy(), r["gts"], r["preds"],
                     f"{r['split']}/{r['file'][:40]}  GT: {len(r['gts'])}  pred: {len(r['preds'])}")
        for ax in np.atleast_1d(axes).ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle("Filled/solid = ground truth (grey = generic damage), dashed = prediction", fontsize=9)
        fig.savefig(out / f"page_{page // args.per_page + 1:02d}.jpg", dpi=110, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(matches)
    df.to_csv(out / "gt_best_iou.csv", index=False)
    df["outcome"] = pd.cut(df["best_iou"], [-0.01, 0.0, 0.1, 0.5, 1.0],
                           labels=["missed (no overlap)", "barely touched (<0.1)", "found, boxed differently (0.1-0.5)", "matched (>=0.5)"])
    n_pred = sum(len(r["preds"]) for r in rows)
    print(f"{len(rows)} images, {len(df)} GT boxes, {n_pred} predictions at conf>={args.conf}, imgsz={imgsz}\n")
    print(pd.crosstab(df["gt_label"], df["outcome"], margins=True))
    print("\nGT box size (px) per label:\n", df.groupby("gt_label")[["gt_w", "gt_h"]].median())
    print(f"\nPages and gt_best_iou.csv written to {out}")


if __name__ == "__main__":
    main()
