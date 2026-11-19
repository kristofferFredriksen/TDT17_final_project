"""Segmentation datasets.

All return (image, mask): image HxWx3 uint8 RGB, mask HxW uint8 with
0 = background, 1 = crack, 2 = pothole (Pothole-Mix) or 0/1 = damage (NRD, RDD2022 boxes).
Use `mask > 0` for binary damage-vs-background.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common.paths import NRD, POTHOLE_MIX, RDD2022

BACKGROUND, CRACK, POTHOLE = 0, 1, 2


def decode_pothole_mix_mask(rgb):
    """Pothole-Mix masks are RGB: green = crack, red = pothole."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[rgb[..., 1] > 127] = CRACK  # threshold, not == 255, in case some masks were resized/antialiased
    mask[rgb[..., 0] > 127] = POTHOLE
    return mask


class PotholeMix:
    def __init__(self, split="training", transform=None, root=POTHOLE_MIX):
        assert split in ("training", "validation", "testing"), split
        self.transform = transform
        self.items = []  # (image_path, mask_path, subset)
        for subset in sorted((Path(root) / split).iterdir()):
            masks = {p.stem: p for p in (subset / "masks").iterdir()}
            for img in sorted((subset / "images").iterdir()):
                if img.stem not in masks:
                    raise FileNotFoundError(f"no mask for {img}")
                self.items.append((img, masks[img.stem], subset.name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, mask_path, _ = self.items[i]
        # Deliberately ignore EXIF orientation: crack500 masks match the raw pixels (verified).
        # Don't add ImageOps.exif_transpose, and use cv2.IMREAD_IGNORE_ORIENTATION if switching to cv2.
        image = Image.open(img_path).convert("RGB")
        rgb = Image.open(mask_path).convert("RGB")
        if image.size != rgb.size:
            raise ValueError(f"size mismatch {image.size} vs {rgb.size}: {img_path}")
        image, mask = np.array(image), decode_pothole_mix_mask(np.array(rgb))
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask


def polygons_to_mask(segmentation, height, width):
    """COCO polygon list -> HxW uint8 0/1 mask."""
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"expected non-empty polygon list, got {type(segmentation).__name__}")  # RLE/empty unsupported
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in segmentation:
        if len(poly) >= 6:  # need at least 3 points
            draw.polygon(poly, fill=1)
    return np.array(canvas)


class NorwegianRoadDamage:
    """All NRD images (train+test merged) as a binary damage eval set.

    The official split leaks (same videos in both), so we use NRD zero-shot (evaluate.py) and
    leave-one-video-out (finetune_nrd.py) instead.
    """

    def __init__(self, transform=None, root=NRD):
        self.transform = transform
        self.items = []  # (image_path, [annotations], height, width)
        for split in ("train", "test"):
            coco = json.loads((Path(root) / split / "_annotations.coco.json").read_text())
            anns = {}
            for a in coco["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            for im in coco["images"]:  # image ids repeat across the two files, so key per file
                self.items.append((Path(root) / split / im["file_name"], anns.get(im["id"], []),
                                   im["height"], im["width"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, anns, h, w = self.items[i]
        image = np.array(Image.open(path).convert("RGB"))
        if image.shape[:2] != (h, w):
            raise ValueError(f"{path}: image {image.shape[:2]} != coco {(h, w)}")
        mask = np.zeros((h, w), np.uint8)
        for a in anns:
            mask |= polygons_to_mask(a["segmentation"], h, w)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask



def rdd_boxes(xml_path, classes):
    """[(xmin, ymin, xmax, ymax)] for objects whose name is in classes, from a Pascal VOC XML file."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        if obj.findtext("name") in classes:
            b = obj.find("bndbox")
            boxes.append(tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
    return boxes


class RDD2022Boxes:
    """RDD2022 detection data as weak segmentation labels: every damage box is filled with 1.

    Only the train folder has labels, so val is carved out of it. The last val_fraction of the
    (sorted) images is used as val instead of a random sample: file numbers likely follow capture
    order, so a random split would put neighbouring frames of the same road in both.
    """
    CLASSES = {"D00", "D10", "D20", "D40"}

    def __init__(self, country="Norway", split="train", val_fraction=0.1, transform=None, root=RDD2022,
                 instance_ids=False):
        assert split in ("train", "val"), split
        base = Path(root) / country / "train"
        images = sorted((base / "images").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no images in {base / 'images'}")
        n_val = max(1, round(len(images) * val_fraction))
        images = images[-n_val:] if split == "val" else images[:-n_val]
        self.items = [(p, base / "annotations" / "xmls" / f"{p.stem}.xml") for p in images]
        self.transform = transform
        self.instance_ids = instance_ids  # True: box k is filled with k (for the box projection loss), else 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, xml_path = self.items[i]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for k, (x0, y0, x1, y1) in enumerate(rdd_boxes(xml_path, self.CLASSES), 1):
            # overlapping boxes: the later one wins the shared pixels
            mask[max(0, int(y0)):min(h, int(np.ceil(y1))), max(0, int(x0)):min(w, int(np.ceil(x1)))] = \
                min(k, 255) if self.instance_ids else 1
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask


if __name__ == "__main__":
    # Sanity check on IDUN: python -m segmentation.dataset
    from collections import Counter

    for split in ("training", "validation", "testing"):
        ds = PotholeMix(split)
        print(split, len(ds), Counter(s for *_, s in ds.items))
    ds = PotholeMix("training")
    for i in range(0, len(ds), max(1, len(ds) // 10)):
        img, m = ds[i]
        print(ds.items[i][2], img.shape, np.unique(m).tolist())

    nrd = NorwegianRoadDamage()
    fg = [(nrd[i][1] > 0).mean() for i in range(len(nrd))]
    print(f"NRD: {len(nrd)} images, damage pixel fraction mean={np.mean(fg):.4f}, empty masks={sum(f == 0 for f in fg)}")
