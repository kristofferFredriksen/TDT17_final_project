"""Box-supervised loss for training segmentation from detection boxes (RDD2022).

Filled boxes overstate damage area (~7x on Norwegian images, Shami et al. 2026), so instead of asking the model to
fill each box we only use what a tight box actually tells us:
  * outside all boxes there is no damage            -> BCE towards 0
  * inside a box we do not know which pixels         -> no pixel-wise loss
  * a tight box touches the damage on every row and every column
                                                     -> projection loss (Tian et al., BoxInst, CVPR 2021):
    the max of the prediction along each row/column of the box must match the box's own projection (all ones).
A thin diagonal crack satisfies the projection term without filling the box.

Not included: BoxInst's pairwise colour-similarity term. Without it, a "+"-shaped prediction would also satisfy
the projections; we rely on the Pothole-Mix pre-training to keep predictions damage-shaped.
"""
import torch
import torch.nn.functional as F


def projection_dice(pred, target, eps=1e-6):
    """1 - Dice between two 1-D projections (BoxInst uses squared terms in the denominator)."""
    return 1 - (2 * (pred * target).sum() + eps) / ((pred ** 2).sum() + (target ** 2).sum() + eps)


def box_projection_loss(logits, box_ids):
    """logits: (B, 1, H, W). box_ids: (B, H, W) with 0 = outside all boxes, k = inside box k."""
    logits = logits[:, 0].float()  # loss in fp32 even under autocast
    box_ids = box_ids.long()
    outside = box_ids == 0
    bg = F.binary_cross_entropy_with_logits(logits[outside], torch.zeros_like(logits[outside])) if outside.any() \
        else logits.sum() * 0
    p = torch.sigmoid(logits)
    terms = []
    for b in range(box_ids.shape[0]):
        for k in box_ids[b].unique():
            if k == 0:
                continue
            m = (box_ids[b] == k).float()
            pk = p[b] * m  # only this box's pixels
            terms.append(projection_dice(pk.amax(1), m.amax(1)) + projection_dice(pk.amax(0), m.amax(0)))
    proj = torch.stack(terms).mean() if terms else logits.sum() * 0
    return bg + proj
