"""Evaluate trained detectors on our RDD2022 test split, per country, and on NorwegianRoadDamage.

Evaluation sets:
  rdd_test           the whole RDD2022 test split from prepare_rdd.py
  rdd_test_<country> the same split, one country at a time
  nrd_agnostic       NorwegianRoadDamage (all 50 images), every annotation counts as "damage",
                     predictions are matched regardless of class. This is the main customer metric,
                     since most NRD annotations have no damage class.
  nrd_perclass       NorwegianRoadDamage with only the D00/D20/D40 annotations. The generic
                     annotations are left out, so predictions on those regions count as false
                     positives: recall is meaningful, precision and mAP are underestimates.

Every row is appended to <out>/results.csv, and Ultralytics' plots (PR curves, confusion
matrix) are saved per model and set.

Usage:
    python detection/evaluate.py --weights /cluster/work/$USER/tdt17/runs/yolo11s_640/weights/best.pt
"""

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

WORK = Path("/cluster/work") / os.environ.get("USER", "") / "tdt17"
NRD_ROOT = "/cluster/projects/vc/courses/TDT17/2026/1_infra/NorwegianRoadDamage"
CLASSES = ["D00", "D10", "D20", "D40"]


def write_yaml(path, image_list, names):
    lines = [f"path: {path.parent}"] + [f"{k}: {image_list}" for k in ("train", "val", "test")]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


def build_rdd_sets(data_dir, sets_dir):
    """Image lists for the whole test split and per country. Labels already exist in data_dir."""
    images = pd.read_csv(data_dir / "images.csv")
    test = images[images["split"] == "test"].copy()
    test["path"] = [str(data_dir / "images" / "test" / Path(p).name) for p in test["img_path"]]

    sets = {"rdd_test": test}
    sets.update({f"rdd_test_{c}": g for c, g in test.groupby("country")})
    yamls = {}
    for name, group in sets.items():
        lst = sets_dir / f"{name}.txt"
        lst.write_text("\n".join(group["path"]) + "\n")
        write_yaml(sets_dir / f"{name}.yaml", lst, CLASSES)
        yamls[name] = sets_dir / f"{name}.yaml"
    return yamls


def nrd_class(name):
    """Map an NRD category name to our class id, or None for the generic 'damage' category."""
    for i, c in enumerate(CLASSES):
        if name.startswith(c):
            return i
    return None


def build_nrd_sets(nrd_root, sets_dir):
    """Convert NorwegianRoadDamage (COCO, train+test) into two YOLO-format evaluation sets."""
    yamls = {}
    for variant in ("nrd_agnostic", "nrd_perclass"):
        img_dir = sets_dir / variant / "images" / "all"
        lbl_dir = sets_dir / variant / "labels" / "all"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "test"):
            coco = json.load(open(Path(nrd_root) / split / "_annotations.coco.json"))
            cats = {c["id"]: c["name"] for c in coco["categories"]}
            lines = {im["id"]: [] for im in coco["images"]}
            sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
            for a in coco["annotations"]:
                cls = 0 if variant == "nrd_agnostic" else nrd_class(cats[a["category_id"]])
                if cls is None:
                    continue
                w, h = sizes[a["image_id"]]
                x, y, bw, bh = a["bbox"]
                lines[a["image_id"]].append(f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")

            for im in coco["images"]:
                src = Path(nrd_root) / split / im["file_name"]
                link = img_dir / im["file_name"]
                if not link.exists():
                    link.symlink_to(src)
                body = "\n".join(lines[im["id"]])
                (lbl_dir / f"{Path(im['file_name']).stem}.txt").write_text(body + ("\n" if body else ""))

        yamls[variant] = sets_dir / f"{variant}.yaml"
        write_yaml(yamls[variant], img_dir, CLASSES)
    return yamls


def evaluate(weights, set_name, yaml, args, out):
    model = YOLO(weights)
    run_name = Path(weights).parent.parent.name  # runs/<name>/weights/best.pt -> <name>
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 640)
    m = model.val(data=str(yaml), split="test", imgsz=imgsz, batch=args.batch,
                  single_cls=set_name == "nrd_agnostic", project=str(out / run_name), name=set_name,
                  exist_ok=True, plots=True, verbose=False)

    row = {"model": run_name, "imgsz": imgsz, "set": set_name,
           "P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map,
           "ms_per_img": sum(m.speed.values())}
    if set_name != "nrd_agnostic":
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            row[f"AP50_{CLASSES[c]}"], row[f"R_{CLASSES[c]}"] = ap50, r
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in row.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--data-dir", default=str(WORK / "rdd_yolo"))
    p.add_argument("--nrd-root", default=NRD_ROOT)
    p.add_argument("--out", default=str(WORK / "eval"))
    p.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sets", nargs="*", default=None, help="only these sets (default: all)")
    args = p.parse_args()

    out = Path(args.out)
    sets_dir = out / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    yamls = build_rdd_sets(Path(args.data_dir), sets_dir) | build_nrd_sets(args.nrd_root, sets_dir)
    if args.sets:
        yamls = {k: v for k, v in yamls.items() if k in args.sets}

    rows = []
    for weights in args.weights:
        for set_name, yaml in yamls.items():
            row = evaluate(weights, set_name, yaml, args, out)
            print(row)
            rows.append(row)

    results = out / "results.csv"
    new = not results.exists()
    fields = ["model", "imgsz", "set", "P", "R", "mAP50", "mAP50-95", "ms_per_img"]
    fields += [f"{m}_{c}" for c in CLASSES for m in ("AP50", "R")]
    with open(results, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nmAP50 per model and set:\n", df.pivot(index="set", columns="model", values="mAP50"))
    print("\nmAP50-95 per model and set:\n", df.pivot(index="set", columns="model", values="mAP50-95"))
    print(f"\nAll rows appended to {results}")


if __name__ == "__main__":
    main()
