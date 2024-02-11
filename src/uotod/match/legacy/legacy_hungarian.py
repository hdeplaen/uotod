from typing import Dict, Optional, Union, Tuple
from warnings import warn

from scipy.optimize import linear_sum_assignment

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ._MatchingMethod import _MatchingMethod
from ..loss.modules import MultipleObjectiveLoss


class HungarianMatching(_MatchingMethod):
    def __init__(self,
                 cls_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 loc_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 bg_class_position: str = "first",
                 bg_cost: float = 10.0,
                 is_anchor_based: bool = False) -> None:
        r"""Hungarian matching method.
        Computes the bipartite matching between the predicted and target boxes using the Hungarian algorithm.
        :param cls_matching_module: Classification loss used to compute the matching, if any.
        :param loc_matching_module: Localization loss used to compute the matching, if any.
        :param bg_class_position: Index of the background class. "first", "last" or "none" (no background class).
        :param bg_cost: Cost of the background class.
        :param is_anchor_based: If True, the matching is performed between the anchor boxes and the target boxes.
        """
        super().__init__(cls_matching_module, loc_matching_module, bg_class_position, bg_cost, is_anchor_based)

    @torch.no_grad()
    def compute_matching(self, cost_matrix: Tensor, target_mask: Tensor) -> Tensor:
        r"""
        Computes the matching between the predicted and target boxes. The matching is computed using the Hungarian
        algorithm.
        :param cost_matrix: the cost matrix. Tensor of shape (batch_size, num_pred, num_tgt + 1).
        :param target_mask: the target mask. Tensor of shape (batch_size, num_tgt).
        :return: the matching. Tensor of shape (batch_size, num_pred, num_tgt + 1). The last entry of the last
            dimension is the background.
        """
        device = cost_matrix.device
        # Move to CPU
        cost_matrix = cost_matrix.cpu()

        # Compute the matching
        matching = torch.zeros_like(cost_matrix)
        for i in range(cost_matrix.shape[0]):
            num_tgt = target_mask[i].sum().item()
            row_ind, col_ind = linear_sum_assignment(cost_matrix[i, :, :num_tgt].numpy())
            matching[i, row_ind, col_ind] = 1.

        # Move back to GPU
        matching = matching.to(device)

        # Assign unmatched predictions to the background
        matching[:, :, -1] = 1. - matching[:, :, :-1].sum(dim=2)

        return matching
