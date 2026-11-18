"""Checks for evaluate.py. Run: python3 -m segmentation.test_evaluate"""
import numpy as np

from segmentation.evaluate import region_hits


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


if __name__ == "__main__":
    test_region_hits()
    print("ok")
