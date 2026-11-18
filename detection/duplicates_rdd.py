"""Near-duplicate check between our RDD2022 splits, using perceptual hashing.

Our split is random at image level (DECISIONS.md, D4), and RDD images are often taken in sequence
along the same drive, so near-identical images can end up in both train and test. That would let
the model "recognise" test images instead of generalising, and inflate the test scores.

Method:
  1. Every image gets a 256-bit difference hash (dHash): the image is shrunk to 17x16 grey pixels
     and each bit says whether a pixel is brighter than its right neighbour. Near-identical images get
     near-identical hashes; the Hamming distance (number of differing bits, 0-256) measures similarity.
  2. For each test and val image, find the closest train image in the same country.
  3. As a reference for "unrelated", also find the closest train image from a *different* country.
     Those can't be duplicates, so that distance distribution shows what chance similarity looks like.
  4. Plot the distributions, count near-duplicates at a few thresholds, and save example pairs so
     the threshold can be checked by eye.

Usage (on IDUN, CPU job, decoding ~36k JPEGs at reduced size takes a few minutes):
    python detection/duplicates_rdd.py
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

WORK = Path("/cluster/work") / os.environ.get("USER", "") / "tdt17"
HASH_W, HASH_H = 16, 16  # 256 bits
THRESHOLDS = [10, 20, 30]  # Hamming distances (out of 256) reported as "near-duplicate"
COUNTRY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def dhash(path):
    try:
        with Image.open(path) as im:
            im.draft("L", (HASH_W * 8, HASH_H * 8))  # let the JPEG decoder skip full resolution
            px = np.asarray(im.convert("L").resize((HASH_W + 1, HASH_H), Image.Resampling.BOX), dtype=np.int16)
        return (px[:, 1:] > px[:, :-1]).ravel()
    except OSError:
        return None


def nearest(query, ref):
    """Hamming distance from each query hash to its closest reference hash, and that reference's index."""
    q, r = query.astype(np.float32), ref.astype(np.float32)
    best_d = np.full(len(q), np.inf, dtype=np.float32)
    best_i = np.zeros(len(q), dtype=np.int64)
    for s in range(0, len(r), 4096):  # chunked to keep memory small
        rc = r[s:s + 4096]
        d = q @ (1 - rc).T + (1 - q) @ rc.T  # number of differing bits
        i = d.argmin(axis=1)
        dm = d[np.arange(len(q)), i]
        better = dm < best_d
        best_d[better], best_i[better] = dm[better], i[better] + s
    return best_d.astype(int), best_i


def pair_page(rows, title, out_file):
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 2.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for (ax_q, ax_r), r in zip(axes, rows):
        for ax, path, label in [(ax_q, r["path"], f"{r['split']}: {r['image']}"),
                                (ax_r, r["nn_path"], f"train: {r['nn_image']}  (distance {r['dist']})")]:
            with Image.open(path) as im:
                im.thumbnail((600, 600))
                ax.imshow(im)
            ax.set_title(label, fontsize=7)
            ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(out_file, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(WORK / "rdd_yolo"))
    p.add_argument("--out", default=str(WORK / "duplicates"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(Path(args.data_dir) / "images.csv")

    cache = out / "hashes.npy"
    if cache.exists():
        hashes = np.load(cache)
    else:
        print(f"Hashing {len(images)} images with {args.workers} processes...", flush=True)
        with Pool(args.workers) as pool:
            hs = pool.map(dhash, images["img_path"], chunksize=64)
        bad = [p for p, h in zip(images["img_path"], hs) if h is None]
        if bad:
            raise SystemExit(f"{len(bad)} images could not be read, e.g. {bad[:3]}")
        hashes = np.stack(hs)
        np.save(cache, hashes)

    rows = []
    for country, group in images.groupby("country"):
        train = group[group["split"] == "train"]
        other_train = images[(images["split"] == "train") & (images["country"] != country)]
        for split in ("val", "test"):
            q = group[group["split"] == split]
            d_same, i_same = nearest(hashes[q.index], hashes[train.index])
            d_other, _ = nearest(hashes[q.index], hashes[other_train.index])
            nn = train.iloc[i_same]
            rows.append(pd.DataFrame({
                "image": q["image"].values, "country": country, "split": split, "path": q["img_path"].values,
                "nn_image": nn["image"].values, "nn_path": nn["img_path"].values,
                "dist": d_same, "dist_other_country": d_other}))
    nn_df = pd.concat(rows, ignore_index=True)
    nn_df.drop(columns=["path", "nn_path"]).to_csv(out / "nearest_train.csv", index=False)

    # Share of val/test images with a near-duplicate in train, per country and threshold
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    test = nn_df[nn_df["split"] == "test"]
    summary = pd.DataFrame({f"dist<={t}": test.groupby("country")["dist"].apply(lambda d: (d <= t).mean())
                            for t in THRESHOLDS})
    summary["median_dist_same_country"] = test.groupby("country")["dist"].median()
    summary["median_dist_other_country"] = test.groupby("country")["dist_other_country"].median()
    summary["min_dist_other_country"] = test.groupby("country")["dist_other_country"].min()
    summary.loc["ALL"] = [(test["dist"] <= t).mean() for t in THRESHOLDS] + [
        test["dist"].median(), test["dist_other_country"].median(), test["dist_other_country"].min()]
    summary.round(3).to_csv(out / "summary_test.csv")
    print("\nTest images: share with a near-duplicate in train (Hamming distance out of 256),\n"
          "and the distance to the closest train image in the same vs another country:\n", summary.round(3))

    # Plot: distance to nearest train image, same country (solid) vs other country (dashed reference)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 130, 4)
    for (country, g), col in zip(test.groupby("country"), COUNTRY_COLORS):
        ax.hist(g["dist"], bins=bins, histtype="step", linewidth=2, density=True, color=col, label=country)
    ax.hist(test["dist_other_country"], bins=bins, histtype="stepfilled", density=True, color="#b5b4ad",
            alpha=0.4, label="reference: nearest image in another country")
    for t in THRESHOLDS:
        ax.axvline(t, color="#52514e", linestyle="--", linewidth=1)
    ax.set_xlabel("Hamming distance from a test image to its nearest train image (0 = identical, 256 bits)")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "nearest_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example pairs: per country the closest test/train pairs, plus pairs around each threshold
    for country, g in test.groupby("country"):
        pair_page(g.nsmallest(4, "dist").to_dict("records"), f"{country}: the 4 closest test/train pairs",
                  out / f"pairs_closest_{country}.jpg")
    band_rows = []
    for lo, hi in [(0, 10), (11, 20), (21, 30), (31, 45)]:
        band = test[(test["dist"] >= lo) & (test["dist"] <= hi)]
        band_rows += band.sample(min(2, len(band)), random_state=args.seed).to_dict("records")
    if band_rows:
        pair_page(band_rows, "Calibration: random pairs at distance 0-10, 11-20, 21-30, 31-45 (top to bottom)",
                  out / "pairs_by_distance.jpg")
    print(f"\nWrote nearest_train.csv, summary_test.csv, nearest_distance.png and example pairs to {out}")


if __name__ == "__main__":
    main()
