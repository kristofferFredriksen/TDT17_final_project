"""Synthetic-data checks for dataset.py. Run: python -m segmentation.test_dataset"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.dataset import CRACK, POTHOLE, NorwegianRoadDamage, PotholeMix, polygons_to_mask


def test_pothole_mix(root):
    sub = root / "training" / "fake"
    (sub / "images").mkdir(parents=True)
    (sub / "masks").mkdir()
    Image.new("RGB", (8, 4)).save(sub / "images/a.jpg")
    rgb = np.zeros((4, 8, 3), np.uint8)
    rgb[0, 0] = (0, 255, 0)
    rgb[1, 1] = (255, 0, 0)
    rgb[2, 2] = (0, 200, 0)  # antialiased crack edge still counts
    Image.fromarray(rgb).save(sub / "masks/a.png")
    img, m = PotholeMix("training", root=root)[0]
    assert img.shape == (4, 8, 3) and m.shape == (4, 8)
    assert m[0, 0] == CRACK and m[1, 1] == POTHOLE and m[2, 2] == CRACK and m.sum() == 2 * CRACK + POTHOLE


def test_nrd(root):
    for split, img_id in (("train", 0), ("test", 0)):  # same id in both files, like the real data
        (root / split).mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(root / split / f"{split}.jpg")
        coco = {"images": [{"id": img_id, "file_name": f"{split}.jpg", "height": 10, "width": 10}],
                "annotations": [{"image_id": img_id, "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]]}] if split == "train" else []}
        (root / split / "_annotations.coco.json").write_text(json.dumps(coco))
    ds = NorwegianRoadDamage(root=root)
    assert len(ds) == 2
    assert ds[0][1].sum() > 0 and set(np.unique(ds[0][1])) == {0, 1}
    assert ds[1][1].sum() == 0  # test image id collision must not pull train annotations


def test_rle_rejected():
    try:
        polygons_to_mask({"counts": [], "size": [2, 2]}, 2, 2)
    except ValueError:
        return
    raise AssertionError("RLE should raise")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_pothole_mix(Path(d) / "pm")
        test_nrd(Path(d) / "nrd")
    test_rle_rejected()
    print("ok")
