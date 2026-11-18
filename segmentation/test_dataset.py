"""Synthetic-data checks for dataset.py. Run: python -m segmentation.test_dataset"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.dataset import CRACK, POTHOLE, NorwegianRoadDamage, PotholeMix, RDD2022Boxes, polygons_to_mask


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


VOC = """<annotation><size><width>20</width><height>10</height></size>{}</annotation>"""
OBJ = "<object><name>{}</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox></object>"


def test_rdd_boxes(root):
    base = root / "Norway" / "train"
    (base / "images").mkdir(parents=True)
    (base / "annotations" / "xmls").mkdir(parents=True)
    objs = {  # D00 box, a non-damage class that must be ignored, a box partly outside the image; one image without boxes
        "Norway_000000": OBJ.format("D00", 2.4, 1, 5.5, 3) + OBJ.format("Repair", 10, 5, 12, 7) + OBJ.format("D40", 18, 8, 25, 12),
        "Norway_000001": "",
        "Norway_000002": OBJ.format("D20", 0, 0, 20, 10),
    }
    for stem, o in objs.items():
        Image.new("RGB", (20, 10)).save(base / "images" / f"{stem}.jpg")
        (base / "annotations" / "xmls" / f"{stem}.xml").write_text(VOC.format(o))
    train = RDD2022Boxes("Norway", "train", val_fraction=0.34, root=root)
    val = RDD2022Boxes("Norway", "val", val_fraction=0.34, root=root)
    assert [p.stem for p, _ in val.items] == ["Norway_000002"] and len(train) == 2  # last image is val
    img, m = train[0]
    assert img.shape == (10, 20, 3) and m.shape == (10, 20)
    assert m[1:3, 2:6].all() and m[1:3, 2:6].sum() == m[:, :12].sum()  # D00 filled (fractional edges rounded out), Repair ignored
    assert m[8:10, 18:20].all()  # clipped D40
    assert train[1][1].sum() == 0 and val[0][1].all()


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
        test_rdd_boxes(Path(d) / "rdd")
    test_rle_rejected()
    print("ok")
