"""Convert RDD2022 (Pascal VOC) into an Ultralytics YOLO dataset.

One pass over the ~38k XML files does three things:
  1. Writes boxes.csv / images.csv with every annotation (all labels, including the
     ones we drop), so EDA can run in seconds without touching the XML files again.
  2. Makes a reproducible train/val/test split (80/10/10), stratified per country.
     The official RDD2022 test folders have no labels, so they are not used.
  3. Writes YOLO labels (class cx cy w h, normalised) for D00/D10/D20/D40 and
     symlinks the images, so the 14 GB of images is not copied.

Usage (on IDUN, run through Slurm, see README):
    python detection/prepare_rdd.py --out /cluster/work/$USER/tdt17/rdd_yolo
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

RDD_ROOT = "/cluster/projects/vc/courses/TDT17/2026/1_infra/RDD2022/RDD2022"
CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def parse_xml(args):
    country, xml_path, img_dir = args
    root = ET.parse(xml_path).getroot()
    name = Path(xml_path).stem
    img_path = Path(img_dir) / root.findtext("filename", default=f"{name}.jpg")
    w = int(float(root.findtext("size/width") or 0))
    h = int(float(root.findtext("size/height") or 0))
    if w <= 0 or h <= 0:  # some VOC files have a missing/zero size, fall back to the image header
        with Image.open(img_path) as im:
            w, h = im.size

    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append({
            "image": name, "country": country, "label": obj.findtext("name").strip(),
            "x1": max(0.0, x1), "y1": max(0.0, y1), "x2": min(w, x2), "y2": min(h, y2),
            "img_w": w, "img_h": h,
        })
    image = {"image": name, "country": country, "img_path": str(img_path), "img_w": w, "img_h": h}
    return image, boxes


def collect_jobs(rdd_root, exclude):
    jobs = []
    for country in sorted(os.listdir(rdd_root)):
        if country in exclude:
            continue
        ann_dir = Path(rdd_root) / country / "train" / "annotations"
        img_dir = Path(rdd_root) / country / "train" / "images"
        for xml_path in ann_dir.rglob("*.xml"):
            jobs.append((country, str(xml_path), str(img_dir)))
    return jobs


def split_images(images, val_frac, test_frac, seed):
    """Random split per country, so every country has the same ratio in each split."""
    rng = random.Random(seed)
    split = {}
    for _, group in images.groupby("country"):
        names = sorted(group["image"])
        rng.shuffle(names)
        n_test, n_val = round(len(names) * test_frac), round(len(names) * val_frac)
        for i, n in enumerate(names):
            split[n] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return images["image"].map(split)


def write_yolo(images, boxes, out):
    kept = boxes[boxes["label"].isin(CLASSES)]
    kept = kept[(kept["x2"] > kept["x1"]) & (kept["y2"] > kept["y1"])]
    by_image = dict(tuple(kept.groupby("image")))

    for row in images.itertuples():
        img_link = out / "images" / row.split / Path(row.img_path).name
        lbl_path = out / "labels" / row.split / f"{Path(row.img_path).stem}.txt"
        img_link.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        if not img_link.exists():
            img_link.symlink_to(row.img_path)

        lines = []
        b = by_image.get(row.image)
        if b is not None:
            for r in b.itertuples():
                cx, cy = (r.x1 + r.x2) / 2 / r.img_w, (r.y1 + r.y2) / 2 / r.img_h
                bw, bh = (r.x2 - r.x1) / r.img_w, (r.y2 - r.y1) / r.img_h
                lines.append(f"{CLASS_ID[r.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # An empty file marks a background image (no damage), which YOLO also trains on.
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (out / "rdd.yaml").write_text("\n".join(yaml) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rdd-root", default=RDD_ROOT)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=["China_Drone"],
                   help="countries to leave out (China_Drone is top-down drone imagery, another domain)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(args.rdd_root, set(args.exclude))
    print(f"Parsing {len(jobs)} XML files with {args.workers} threads...")
    # Threads, not processes: the bottleneck is network-filesystem latency, not CPU.
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(parse_xml, jobs))

    images = pd.DataFrame([r[0] for r in results])
    boxes = pd.DataFrame([b for r in results for b in r[1]])
    images["split"] = split_images(images, args.val_frac, args.test_frac, args.seed)
    boxes = boxes.merge(images[["image", "split"]], on="image")

    images.to_csv(out / "images.csv", index=False)
    boxes.to_csv(out / "boxes.csv", index=False)
    write_yolo(images, boxes, out)

    kept = boxes[boxes["label"].isin(CLASSES)]
    print(f"\n{len(images)} images, {len(kept)} kept boxes, {len(boxes) - len(kept)} dropped boxes")
    print(pd.crosstab(kept["split"], kept["label"], margins=True))
    print(pd.crosstab(images["country"], images["split"], margins=True))
    print(f"\nWrote {out / 'rdd.yaml'}")


if __name__ == "__main__":
    main()
