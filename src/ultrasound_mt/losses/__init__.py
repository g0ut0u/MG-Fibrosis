"""Loss functions.

- ``dice_loss_with_logits`` / ``MultiTaskLoss``: multi-task (segmentation Dice + classification Focal)
- ``WeightedFocalWithLogitsLoss``: weighted Focal classification loss
"""
from .multi_task import (
    dice_loss_with_logits,
    WeightedFocalWithLogitsLoss,
    MultiTaskLoss,
)

__all__ = [
    "MultiTaskLoss",
    "WeightedFocalWithLogitsLoss",
    "dice_loss_with_logits",
]
