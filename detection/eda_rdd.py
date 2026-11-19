"""Exploratory data analysis of RDD2022, based on the CSVs written by prepare_rdd.py.

Reads only boxes.csv / images.csv, so it runs in seconds. Plots are written as PNGs
for the presentation, summary tables as CSV, and the key numbers are printed.

Usage:
    python detection/eda_rdd.py --data /cluster/work/$USER/tdt17/rdd_yolo --out eda_out
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = {"D00": "D00 longitudinal", "D10": "D10 transverse", "D20": "D20 alligator", "D40": "D40 pothole"}
CLASS_COLORS = {"D00": "#2a78d6", "D10": "#eb6834", "D20": "#1baf7a", "D40": "#eda100"}
NEUTRAL = "#52514e"
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")  # 10 instead of 10^1 on log axes
TRAIN_SIZE = 640  # default YOLO input size; used to show how small boxes become after resizing

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": NEUTRAL, "ytick.color": NEUTRAL,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"  saved {name}")


def plot_class_per_country(kept, out):
    counts = pd.crosstab(kept["country"], kept["label"]).reindex(columns=CLASSES, fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    share = counts.div(counts.sum(axis=1), axis=0)

    for data, name, xlabel in [(counts, "class_per_country.png", "Number of boxes"),
                               (share, "class_share_per_country.png", "Share of boxes in country")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        left = np.zeros(len(data))
        for c in CLASSES:
            ax.barh(data.index, data[c], left=left, color=CLASS_COLORS[c], label=CLASS_NAMES[c],
                    height=0.6, edgecolor="white", linewidth=2)
            left += data[c].values
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, out, name)
    return counts


def plot_background(images, kept, out):
    has_damage = images["image"].isin(kept["image"])
    bg = (~has_damage).groupby(images["country"]).mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(bg.index, bg.values, color=CLASS_COLORS["D00"], height=0.6)
    for y, v in enumerate(bg.values):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=NEUTRAL)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of images with no D00/D10/D20/D40 box (background images)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "background_per_country.png")
    return bg


def plot_box_size(kept, out):
    # sqrt of the box area relative to the image area: 0.1 means the box is "10% of the image side".
    rel = np.sqrt((kept["w"] * kept["h"]) / (kept["img_w"] * kept["img_h"]))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.logspace(np.log10(0.005), 0, 50)
    for c in CLASSES:
        ax.hist(rel[kept["label"] == c], bins=bins, histtype="step", linewidth=2, density=True,
                color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("Relative box size  sqrt(box area / image area)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, out, "box_size_per_class.png")


def plot_box_px_after_resize(kept, out):
    # Shortest box side in pixels once the image is resized so its long side is TRAIN_SIZE.
    scale = TRAIN_SIZE / kept[["img_w", "img_h"]].max(axis=1)
    short = kept[["w", "h"]].min(axis=1) * scale
    order = short.groupby(kept["country"]).median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot([short[kept["country"] == c] for c in order], orientation="horizontal", tick_labels=order,
               showfliers=False, widths=0.5, medianprops={"color": CLASS_COLORS["D10"], "linewidth": 2})
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.axvline(8, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(8.3, len(order) + 0.3, "8 px", color=NEUTRAL)
    ax.set_xlabel(f"Shortest box side in pixels after resizing to {TRAIN_SIZE}px")
    ax.grid(axis="y", visible=False)
    save(fig, out, f"box_px_at_{TRAIN_SIZE}_per_country.png")
    return (short < 8).groupby(kept["country"]).mean()


def plot_aspect_ratio(kept, out):
    ar = np.log2(kept["w"] / kept["h"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(-5, 5, 60)
    for c in CLASSES:
        ax.hist(ar[kept["label"] == c], bins=bins, histtype="step", linewidth=2, density=True,
                color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_xticks(range(-4, 5, 2), ["1:16", "1:4", "1:1", "4:1", "16:1"])
    ax.set_xlabel("Box aspect ratio (width : height)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, out, "aspect_ratio_per_class.png")


def plot_box_centers(kept, out):
    cx = (kept["x1"] + kept["x2"]) / 2 / kept["img_w"]
    cy = (kept["y1"] + kept["y2"]) / 2 / kept["img_h"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, c in zip(axes, CLASSES):
        m = kept["label"] == c
        ax.hist2d(cx[m], cy[m], bins=30, range=[[0, 1], [0, 1]], cmap="Blues")
        ax.set_title(CLASS_NAMES[c])
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Where in the image the boxes are (darker = more boxes)", y=1.02)
    save(fig, out, "box_centers_per_class.png")


def plot_examples(images, kept, out, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    paths = images.set_index("image")["img_path"]
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(3 * per_class, 3 * len(CLASSES)))
    for row, c in zip(axes, CLASSES):
        # One image per country where possible, so the grid also shows the domain differences.
        cands = kept[kept["label"] == c].drop_duplicates("image")
        picks = cands.groupby("country").sample(1, random_state=seed)
        picks = picks.sample(min(per_class, len(picks)), random_state=seed)
        for ax, r in zip(row, picks.itertuples()):
            with Image.open(paths[r.image]) as im:
                im.thumbnail((800, 800))
                s = im.width / r.img_w
                ax.imshow(im)
            for b in kept[(kept["image"] == r.image) & (kept["label"] == c)].itertuples():
                ax.add_patch(plt.Rectangle((b.x1 * s, b.y1 * s), b.w * s, b.h * s, fill=False,
                                           edgecolor=CLASS_COLORS[c], linewidth=2))
            ax.set_title(f"{CLASS_NAMES[c]}\n{r.country}", fontsize=8)
        for ax in row:
            ax.axis("off")
    save(fig, out, "examples_per_class.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with boxes.csv / images.csv from prepare_rdd.py")
    p.add_argument("--out", default="eda_out")
    p.add_argument("--no-examples", action="store_true", help="skip the example grid (reads images)")
    args = p.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(data / "images.csv")
    boxes = pd.read_csv(data / "boxes.csv")
    boxes["w"], boxes["h"] = boxes["x2"] - boxes["x1"], boxes["y2"] - boxes["y1"]
    kept = boxes[boxes["label"].isin(CLASSES) & (boxes["w"] > 0) & (boxes["h"] > 0)]

    print("Plots:")
    counts = plot_class_per_country(kept, out)
    bg = plot_background(images, kept, out)
    plot_box_size(kept, out)
    tiny = plot_box_px_after_resize(kept, out)
    plot_aspect_ratio(kept, out)
    plot_box_centers(kept, out)
    if not args.no_examples:
        plot_examples(images, kept, out)

    summary = counts.copy()
    summary["images"] = images.groupby("country").size()
    summary["background_share"] = bg.round(3)
    summary["boxes_per_damaged_image"] = (kept.groupby("country").size()
                                          / kept.groupby("country")["image"].nunique()).round(2)
    summary[f"share_boxes_under_8px_at_{TRAIN_SIZE}"] = tiny.round(3)
    summary.to_csv(out / "summary_per_country.csv")

    dropped = boxes[~boxes["label"].isin(CLASSES)]
    dropped_tab = pd.crosstab(dropped["label"], dropped["country"], margins=True)
    dropped_tab.to_csv(out / "dropped_labels.csv")

    split_tab = pd.crosstab(kept["split"], kept["label"], normalize="index").round(3)
    split_tab.to_csv(out / "class_share_per_split.csv")

    pd.set_option("display.width", 200)
    print("\nPer country:\n", summary)
    print("\nDropped labels:\n", dropped_tab)
    print("\nClass share per split (should be similar across splits):\n", split_tab)


if __name__ == "__main__":
    main()
