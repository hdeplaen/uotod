import torch.nn as nn

import uotod
from sample import input, target, img, anchors


matching_method = uotod.match.UnbalancedSinkhorn(
    cls_matching_module=None,                                   # No classification cost
    loc_matching_module=uotod.loss.IoULoss(reduction="none"),   # Use IoU as localization cost
    bg_cost=0.8,                                                # Threshold for matching to background
    is_anchor_based=True,                                       # Use anchor-based matching
    reg_pred=1e+3,
    reg_target=1e-3,
    reg0=0.03,
)

criterion = uotod.loss.DetectionLoss(
    matching_method=matching_method,
    cls_loss_module=nn.CrossEntropyLoss(reduction="none"),
    loc_loss_module=nn.L1Loss(reduction="none"),
)

loss_value = criterion(input, target, anchors)
