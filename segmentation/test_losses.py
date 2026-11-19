"""Checks for the box projection loss. Run: python3 -m segmentation.test_losses"""
import numpy as np
import torch

from segmentation.losses import box_projection_loss
from segmentation.train import Transform

BIG = 12.0  # logit far from 0: sigmoid ~ 1 or ~ 0


def logits_from(pred):
    return (pred.float() * 2 * BIG - BIG)[None, None]  # (1, 1, H, W): True -> +BIG, False -> -BIG


def test_projection_loss():
    ids = torch.zeros(1, 20, 20, dtype=torch.long)
    ids[0, 4:14, 4:14] = 1                                   # one 10x10 box
    diag = torch.zeros(20, 20, dtype=torch.bool)
    diag[range(4, 14), range(4, 14)] = True                  # thin crack along the box diagonal
    filled = ids[0] == 1
    empty = torch.zeros(20, 20, dtype=torch.bool)
    outside = empty.clone(); outside[16:19, 0:10] = True     # damage predicted outside every box
    half = torch.zeros(20, 20, dtype=torch.bool); half[4:9, 4:14] = True  # covers only the top half of the box

    loss = lambda pred: box_projection_loss(logits_from(pred), ids).item()
    assert loss(diag) < 1e-3, loss(diag)                     # a crack that touches every row/column is NOT penalised
    assert loss(filled) < 1e-3                               # neither is filling the box (inside is "unknown")
    assert loss(empty) > 0.9                                 # predicting nothing in the box is penalised
    assert loss(half) > 0.1                                  # missing half the rows is penalised
    assert loss(diag | outside) > loss(diag) + 1e-3          # false positives outside boxes are penalised


def test_no_boxes_and_two_boxes():
    ids = torch.zeros(1, 8, 8, dtype=torch.long)
    assert box_projection_loss(logits_from(torch.zeros(8, 8, dtype=torch.bool)), ids).item() < 1e-3
    ids[0, 0:3, 0:3] = 1; ids[0, 5:8, 5:8] = 2
    only_first = torch.zeros(8, 8, dtype=torch.bool); only_first[range(3), range(3)] = True
    assert box_projection_loss(logits_from(only_first), ids).item() > 0.9  # second box unexplained: mean over boxes


def test_gradients_flow():
    ids = torch.zeros(2, 16, 16, dtype=torch.long); ids[0, 2:10, 3:12] = 1
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    box_projection_loss(logits, ids).backward()
    assert logits.grad.abs().sum() > 0 and torch.isfinite(logits.grad).all()


def test_transform_keeps_box_ids():
    mask = np.zeros((40, 60), np.uint8); mask[5:15, 5:15] = 1; mask[20:30, 30:50] = 2
    out = Transform(30, binary=False)(np.zeros((40, 60, 3), np.uint8), mask)
    assert set(out["mask"].unique().tolist()) == {0, 1, 2}   # ids survive the (nearest) resize
    assert Transform(30)(np.zeros((40, 60, 3), np.uint8), mask)["mask"].max() == 1


if __name__ == "__main__":
    test_projection_loss()
    test_no_boxes_and_two_boxes()
    test_gradients_flow()
    test_transform_keeps_box_ids()
    print("ok")
