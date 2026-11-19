"""Checks for train.py helpers. Run: python3 -m segmentation.test_train"""
import numpy as np
import torch

from segmentation.train import Transform, evaluate, predict


def test_transform_downscale_only():
    t = Transform(max_long_side=100)
    big = t(np.zeros((50, 200, 3), np.uint8), np.full((50, 200), 2, np.uint8))
    assert big["image"].shape == (3, 25, 100) and big["mask"].max() == 1  # class 2 -> binary 1
    small = t(np.zeros((40, 60, 3), np.uint8), np.zeros((40, 60), np.uint8))
    assert small["image"].shape == (3, 40, 60)  # never upsampled


def test_predict_crops_back():
    model = torch.nn.Conv2d(3, 1, 1)
    assert predict(model, torch.zeros(1, 3, 37, 50)).shape == (1, 1, 37, 50)


def test_evaluate():
    y = torch.zeros(1, 4, 4); y[0, :2] = 1  # 8 positive pixels

    class Fixed(torch.nn.Module):
        def __init__(self, pred): super().__init__(); self.pred = pred
        def forward(self, x): return (self.pred.float() * 20 - 10).unsqueeze(1)  # logits

    perfect = evaluate(Fixed(y.bool()), [(torch.zeros(1, 3, 4, 4), y)], "cpu")
    assert abs(perfect["iou"] - 1) < 1e-6 and abs(perfect["dice"] - 1) < 1e-6
    half = y.clone(); half[0, 1] = 0  # predict 4 of 8 positives, no false positives
    m = evaluate(Fixed(half.bool()), [(torch.zeros(1, 3, 4, 4), y)], "cpu")
    assert abs(m["iou"] - 0.5) < 1e-6 and abs(m["precision"] - 1) < 1e-6 and abs(m["recall"] - 0.5) < 1e-6
    assert abs(m["miou_img"] - 0.5) < 1e-6


if __name__ == "__main__":
    test_transform_downscale_only()
    test_predict_crops_back()
    test_evaluate()
    print("ok")
