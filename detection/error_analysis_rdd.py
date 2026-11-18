"""Error analysis of a detector on our RDD2022 test split, with a visual check per country.

Predicts once on the whole test split (conf >= 0.25, the operating point a user would see) and
matches every ground-truth box to its best-overlapping prediction (IoU, class ignored). Each
ground-truth box gets one outcome:
  missed              no prediction overlaps (IoU < 0.1)
  boxed differently   0.1 <= IoU < 0.5
  wrong class         IoU >= 0.5 but another class
  correct             IoU >= 0.5 and the right class
Predictions that overlap no ground truth (IoU < 0.1) are counted as false alarms.
Matching is per box (not one-to-one like mAP), which is fine for describing error types.

Outputs: outcome tables per country and class, found-rate by box size (at the model's input
size), false alarms per image, and one page of example images per country.

Usage (on IDUN, GPU, a few minutes):
    python detection/error_analysis_rdd.py --weights /cluster/work/$USER/tdt17/runs/yolo11s_640/weights/best.pt
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from visualize_nrd import COLORS, iou

WORK = Path("/cluster/work") / os.environ.get("USER", "") / "tdt17"
CLASSES = ["D00", "D10", "D20", "D40"]
OUTCOMES = ["missed", "boxed differently", "wrong class", "correct"]
SIZE_BINS = [0, 8, 16, 32, 64, 1e9]
SIZE_LABELS = ["<8", "8-16", "16-32", "32-64", ">=64"]
COUNTRY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def outcome(best_iou, same_class):
    if best_iou < 0.1:
        return "missed"
    if best_iou < 0.5:
        return "boxed differently"
    return "correct" if same_class else "wrong class"


def predict_all(model, paths, imgsz, conf, batch):
    preds = {}
    for res in model.predict(paths, imgsz=imgsz, conf=conf, batch=batch, stream=True, verbose=False):
        preds[Path(res.path).stem] = [
            {"label": CLASSES[int(c)], "conf": float(s), "box": tuple(b)}
            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
    return preds


def draw_page(ax, img, gts, preds, title):
    ax.imshow(img)
    for g in gts:
        x1, y1, x2, y2 = g["box"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=COLORS[g["label"]], linewidth=2))
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=COLORS[p["label"]],
                                   linewidth=1.5, linestyle="--"))
        ax.text(x1, y1 - 4, f"{p['label']} {p['conf']:.2f}", color="white", fontsize=6,
                bbox={"facecolor": COLORS[p["label"]], "edgecolor": "none", "pad": 1})
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data-dir", default=str(WORK / "rdd_yolo"))
    p.add_argument("--out", default=None, help="default: <WORK>/error_analysis/<run name>")
    p.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--per-country", type=int, default=6, help="example images per country")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_name = Path(args.weights).parent.parent.name
    out = Path(args.out or WORK / "error_analysis" / run_name)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    model = YOLO(args.weights)
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 640)

    images = pd.read_csv(data / "images.csv")
    images = images[images["split"] == "test"].copy()
    images["path"] = [str(data / "images" / "test" / Path(p).name) for p in images["img_path"]]
    boxes = pd.read_csv(data / "boxes.csv")
    boxes = boxes[(boxes["split"] == "test") & boxes["label"].isin(CLASSES)
                  & (boxes["x2"] > boxes["x1"]) & (boxes["y2"] > boxes["y1"])].copy()

    print(f"Predicting {len(images)} test images at imgsz={imgsz}, conf>={args.conf}...")
    preds = predict_all(model, images["path"].tolist(), imgsz, args.conf, args.batch)

    # Ground-truth outcomes
    gt_rows = []
    for b in boxes.itertuples():
        gt = (b.x1, b.y1, b.x2, b.y2)
        cands = preds.get(Path(b.image).stem, [])
        best_iou, best = max([(iou(gt, q["box"]), q) for q in cands], key=lambda t: t[0], default=(0.0, None))
        scale = imgsz / max(b.img_w, b.img_h)
        gt_rows.append({"image": b.image, "country": b.country, "label": b.label,
                        "short_px": min(b.x2 - b.x1, b.y2 - b.y1) * scale, "best_iou": best_iou,
                        "outcome": outcome(best_iou, best is not None and best["label"] == b.label),
                        "pred_label": best["label"] if best else ""})
    gt_df = pd.DataFrame(gt_rows)
    gt_df["size"] = pd.cut(gt_df["short_px"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    gt_df.to_csv(out / "gt_outcomes.csv", index=False)

    # False alarms: predictions overlapping no ground-truth box
    gts_by_image = {k: list(zip(g.x1, g.y1, g.x2, g.y2)) for k, g in boxes.groupby("image")}
    fa_rows = []
    for im in images.itertuples():
        gts = gts_by_image.get(im.image, [])
        for q in preds.get(Path(im.path).stem, []):
            if max((iou(q["box"], g) for g in gts), default=0.0) < 0.1:
                fa_rows.append({"image": im.image, "country": im.country, "label": q["label"],
                                "conf": q["conf"], "background_image": not gts})
    fa_df = pd.DataFrame(fa_rows, columns=["image", "country", "label", "conf", "background_image"])
    fa_df.to_csv(out / "false_alarms.csv", index=False)

    # Tables
    pd.set_option("display.width", 200)
    by_country = pd.crosstab(gt_df["country"], gt_df["outcome"], normalize="index").reindex(columns=OUTCOMES, fill_value=0)
    by_class = pd.crosstab(gt_df["label"], gt_df["outcome"], normalize="index").reindex(columns=OUTCOMES, fill_value=0)
    gt_df["found"] = gt_df["best_iou"] >= 0.5
    by_size = gt_df.pivot_table(index="size", columns="country", values="found", aggfunc="mean", observed=False).reindex(SIZE_LABELS)
    size_counts = pd.crosstab(gt_df["size"], gt_df["country"]).reindex(SIZE_LABELS, fill_value=0)
    wrong = gt_df[gt_df["outcome"] == "wrong class"]
    confusion = pd.crosstab(wrong["label"], wrong["pred_label"]).rename_axis(index="true", columns="predicted")
    fa_per_img = (fa_df.groupby("country").size() / images.groupby("country").size()).fillna(0)
    fa_bg = (fa_df[fa_df["background_image"]].groupby("country").size()
             / images[~images["image"].isin(boxes["image"])].groupby("country").size()).fillna(0)
    false_alarms = pd.DataFrame({"per_image": fa_per_img.round(2), "per_background_image": fa_bg.round(2)})

    for name, df in [("outcome_by_country", by_country), ("outcome_by_class", by_class),
                     ("found_by_size", by_size), ("gt_count_by_size", size_counts),
                     ("wrong_class_confusion", confusion), ("false_alarms_per_image", false_alarms)]:
        df.round(3).to_csv(out / f"{name}.csv")
        print(f"\n{name}:\n{df.round(3)}")

    # Plot: found-rate by box size, one line per country
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for c, col in zip(by_size.columns, COUNTRY_COLORS):
        ax.plot(SIZE_LABELS, by_size[c].values, marker="o", markersize=6, linewidth=2, color=col, label=c)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Shortest side of the true box, in pixels at the model input ({imgsz} px)")
    ax.set_ylabel("Share found (IoU >= 0.5)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out / "found_by_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example pages: per country, the images with the most damage among a random sample
    for country, group in images.groupby("country"):
        has_gt = group[group["image"].isin(boxes["image"])]
        if has_gt.empty:
            continue
        sample = has_gt.sample(min(len(has_gt), args.per_country * 4), random_state=args.seed)
        sample = sample.assign(n=sample["image"].map(boxes.groupby("image").size()).astype(int)).nlargest(args.per_country, "n")
        fig, axes = plt.subplots((len(sample) + 1) // 2, 2, figsize=(12, 4.2 * ((len(sample) + 1) // 2)))
        for ax, im in zip(np.atleast_1d(axes).ravel(), sample.itertuples()):
            g = boxes[boxes["image"] == im.image]
            gts = [{"label": r.label, "box": (r.x1, r.y1, r.x2, r.y2)} for r in g.itertuples()]
            with Image.open(im.path) as pic:
                draw_page(ax, pic.copy(), gts, preds.get(Path(im.path).stem, []),
                          f"{im.image}  GT: {len(gts)}  pred: {len(preds.get(Path(im.path).stem, []))}")
        for ax in np.atleast_1d(axes).ravel()[len(sample):]:
            ax.axis("off")
        fig.suptitle(f"{country}: solid = ground truth, dashed = prediction (conf >= {args.conf})", fontsize=9)
        fig.savefig(out / f"examples_{country}.jpg", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"\nWrote tables, found_by_size.png and examples_<country>.jpg to {out}")


if __name__ == "__main__":
    main()
