"""Checks for evaluate.py. Run: python3 -m segmentation.test_evaluate"""
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from segmentation.evaluate import append_row, nrd_metrics, region_hits


def test_region_hits():
    gt = np.zeros((10, 20), bool)
    gt[1:5, 1:5] = True      # region A: 16 px
    gt[1:5, 10:15] = True    # region B: 20 px
    pred = np.zeros_like(gt)
    pred[2, 2] = True        # 1/16 = 6% of A -> hit at 5% threshold
    pred[8, 18] = True       # false positive outside any region: must not count
    assert region_hits(gt, pred) == (2, 1)
    assert region_hits(gt, pred, min_cover=0.1) == (2, 0)
    assert region_hits(np.zeros_like(gt), pred) == (0, 0)


def test_nrd_metrics_per_image_and_area(tmp):
    g1 = np.zeros((4, 5), bool); g1[0, :4] = True          # 4 px damage
    p1 = np.zeros_like(g1); p1[0, :2] = True                 # 2 px predicted, IoU 0.5
    g2 = np.zeros_like(g1); p2 = np.zeros_like(g1); p2[3, :] = True  # empty GT, 5 px false positive
    x = torch.zeros(3, 4, 5)
    m = nrd_metrics([(x, g1, p1), (x, g2, p2)], [], tmp / "fig.png")
    assert abs(m["miou_img"] - 0.5) < 1e-9                   # only images with damage count
    assert abs(m["iou"] - 2 / 9) < 1e-9                      # micro: tp 2, fp 5, fn 2
    assert abs(m["area_pct_gt"] - 10) < 1e-9 and abs(m["area_pct_pred"] - 17.5) < 1e-9
    assert abs(m["area_pct_mae"] - 17.5) < 1e-9              # (|10-20| + |25-0|) / 2


def test_append_row_adds_columns(tmp):
    t = tmp / "r.csv"
    append_row(t, {"a": 1, "b": 2})
    append_row(t, {"a": 3, "c": 4})
    rows = list(csv.DictReader(open(t)))
    assert list(rows[0]) == ["a", "b", "c"] and rows[0]["c"] == "" and rows[1] == {"a": "3", "b": "", "c": "4"}


if __name__ == "__main__":
    test_region_hits()
    with tempfile.TemporaryDirectory() as d:
        test_nrd_metrics_per_image_and_area(Path(d))
        test_append_row_adds_columns(Path(d))
    print("ok")
