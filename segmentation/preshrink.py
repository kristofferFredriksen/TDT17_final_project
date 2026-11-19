"""Write a copy of Pothole-Mix with every image downscaled to at most 1024 px on the long side.

Training caps images at max_long_side=1024 anyway, so the model sees the same input, but each epoch decodes
far smaller files. Same resize as segmentation.train.Transform (cv2 INTER_AREA for images, INTER_NEAREST for masks).
Images that are already small enough are copied byte-for-byte; resized JPEGs are saved at quality 95 (near-lossless),
PNGs stay lossless. NRD is only linked, not copied.

    python3 -m segmentation.preshrink /cluster/work/$USER/tdt17/data_1024
    DATA_ROOT=/cluster/work/$USER/tdt17/data_1024 python3 -m segmentation.train ...
"""
import argparse
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common.paths import DATA_ROOT, NRD, POTHOLE_MIX

MARKER = "preshrunk.json"


def target_size(w, h, long_side):
    s = long_side / max(w, h)
    return None if s >= 1 else (round(w * s), round(h * s))  # same rounding as Transform


def shrink(job):
    src, dst, long_side, is_mask = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src)  # PIL does not apply EXIF rotation, matching dataset.py (crack500 masks follow raw pixels)
    size = target_size(*im.size, long_side)
    if size is None:
        shutil.copy2(src, dst)
        return
    arr = np.array(im.convert("RGB"))
    arr = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    out = Image.fromarray(arr)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        out.save(dst, quality=95)
    else:
        out.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", type=Path, help="new DATA_ROOT, e.g. /cluster/work/$USER/tdt17/data_1024")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    args = ap.parse_args()

    out_pm = args.out_root / POTHOLE_MIX.relative_to(DATA_ROOT)
    jobs = [(src, out_pm / src.relative_to(POTHOLE_MIX), args.long_side, src.parent.name == "masks")
            for src in sorted(POTHOLE_MIX.glob("*/*/*/*")) if src.parent.name in ("images", "masks")]
    print(f"{len(jobs)} files -> {out_pm}")
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(shrink, jobs, chunksize=16), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    for extra in POTHOLE_MIX.glob("*.txt"):
        shutil.copy2(extra, out_pm / extra.name)
    nrd_link = args.out_root / NRD.relative_to(DATA_ROOT)
    if not nrd_link.exists():
        nrd_link.symlink_to(NRD.resolve(), target_is_directory=True)
    (args.out_root / MARKER).write_text(json.dumps({"max_long_side": args.long_side, "source": str(POTHOLE_MIX)}))
    print("done")


if __name__ == "__main__":
    main()
