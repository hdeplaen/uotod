from abc import ABCMeta, ABC, abstractmethod
from typing import Dict, Union, List, Optional
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ..loss.modules import MultipleObjectiveLoss
from ..utils import convert_target_to_dict


class _MatchingMethod(nn.Module):
    def __init__(self,
                 cls_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 loc_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 bg_class_position: str = "first",
                 bg_cost: float = 10.0,
                 is_anchor_based: bool = False) -> None:
        r"""Matching method. This class is an abstract class. All other matching methods must inherit from this class.
        :param cls_matching_module: Classification loss used to compute the matching, if any.
        :param loc_matching_module: Localization loss used to compute the matching, if any.
        :param bg_class_position: Index of the background class. "first", "last" or "none" (no background class).
        :param bg_cost: Cost of the background class.
        :param is_anchor_based: If True, the matching is performed between the anchor boxes and the target boxes.
        """
        super().__init__()

        assert cls_matching_module is not None or loc_matching_module is not None, \
            "At least one of the classification and localization losses must be provided."
        assert bg_cost >= 0., "The background cost must be non-negative."
        assert bg_class_position in ["first", "last", "none"], \
            "bg_class_index must be 'first', 'last' or 'none'"

        self.matching_cls_module = cls_matching_module if cls_matching_module is not None else MultipleObjectiveLoss()
        self.matching_loc_module = loc_matching_module if loc_matching_module is not None else MultipleObjectiveLoss()
        self.bg_class_position = bg_class_position  # FIXME: this is not used; but do we need it?
        self.bg_cost = bg_cost
        self.is_anchor_based = is_anchor_based

    @torch.no_grad()
    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes a batch of matchings between the predicted and target boxes.
        :param input: dictionary containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :param target: dictionary containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :param anchors: the anchors used to compute the predicted boxes.
            Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :return: the matching between the predicted and target boxes:
            Tensor of shape (batch_size, num_pred, num_targets).
        """
        if self.is_anchor_based:
            assert anchors is not None, "The anchors must be provided for anchor-based matching methods."
        elif anchors is not None:
            warn("The anchors are provided but the matching method is not anchor-based.")

        # Convert the target to dict of masked tensors if necessary.
        if not isinstance(target, dict):
            target = convert_target_to_dict(target)

        # Compute the cost matrix.
        cls_cost = self._compute_cls_costs(input["pred_logits"], target["labels"])
        if self.is_anchor_based:
            loc_cost = self._compute_loc_costs(anchors, target["boxes"])
        else:
            loc_cost = self._compute_loc_costs(input["pred_boxes"], target["boxes"])
        cost_matrix = cls_cost + loc_cost

        # Mask the cost matrix.
        cost_matrix = cost_matrix * target["mask"].unsqueeze(dim=1).expand(cost_matrix.shape)

        # Add the background cost.
        cost_matrix = torch.cat([cost_matrix, self.bg_cost * torch.ones_like(cost_matrix[:, :, :1])], dim=2)

        # Compute the matching.
        matching = self.compute_matching(cost_matrix, target["mask"])

        # Verify that the matching is not NaN.
        if torch.isnan(matching).any():
            raise RuntimeError("The matching is NaN, this may be caused by a too low regularization parameter.")

        return matching

    def _compute_cls_costs(self, pred_logits: Tensor, tgt_classes: Tensor) -> Tensor:
        r"""
        Computes the classification cost matrix.
        :param pred_logits: Predicted logits. Tensor of shape (batch_size, num_pred, num_classes).
        :param tgt_classes: Target classes. Tensor of shape (batch_size, num_targets).
        :return: the classification cost matrix. Tensor of shape (batch_size, num_pred, num_classes).
        """
        # TODO: check if the function works OK for DETR, DEf-DETR, and SSD
        batch_size, num_pred, num_classes = pred_logits.shape
        num_tgt = tgt_classes.shape[1]
        is_onehot = (len(tgt_classes.shape) == 3)

        pred_logits_rep = pred_logits.unsqueeze(dim=2).repeat(1, 1, num_tgt, 1).view(
            batch_size * num_pred * num_tgt, -1)

        if is_onehot:
            tgt_classes_rep = tgt_classes.unsqueeze(dim=1).repeat(1, num_pred, 1, 1)
            tgt_classes_rep = tgt_classes_rep.view(batch_size * num_pred * num_tgt, num_classes)
        else:
            tgt_classes_rep = tgt_classes.unsqueeze(dim=1).repeat(1, num_pred, 1)
            tgt_classes_rep = tgt_classes_rep.view(batch_size * num_pred * num_tgt)

        # Compute the classification cost matrix.
        cls_cost = self.matching_cls_module(pred_logits_rep, tgt_classes_rep).view(batch_size, num_pred, num_tgt)

        return cls_cost

    def _compute_loc_costs(self, pred_locs: Tensor, tgt_locs: Tensor) -> Tensor:
        r"""
        Computes the localization cost matrix.
        :param pred_locs: Predicted locations. Tensor of shape (batch_size, num_pred, 4),
        :param tgt_locs: Target locations. Tensor of shape (batch_size, num_targets, 4),
        :return: the localization cost matrix. Tensor of shape (batch_size, num_pred, num_targets).
        """
        batch_size, num_pred = pred_locs.shape[:2]
        num_tgt = tgt_locs.shape[1]

        pred_locs_rep = pred_locs.unsqueeze(dim=2).repeat(1, 1, num_tgt, 1).view(
            batch_size * num_pred * num_tgt, 4)
        tgt_locs_rep = tgt_locs.unsqueeze(dim=1).repeat(1, num_pred, 1, 1).view(
            batch_size * num_pred * num_tgt, 4)

        # Compute the localization cost matrix.
        loc_cost = self.matching_loc_module(pred_locs_rep, tgt_locs_rep).view(batch_size, num_pred, num_tgt)

        return loc_cost

    @abstractmethod
    def compute_matching(self, cost_matrix: Tensor, target_mask: Tensor) -> Tensor:
        r"""
        Computes the matching.
        :param cost_matrix: Cost matrix. Tensor of shape (batch_size, num_pred, num_targets + 1).
        :param target_mask: Target mask. Tensor of shape (batch_size, num_targets).
        :return: the matching. Tensor of shape (batch_size, num_pred, num_targets + 1). The last entry of the last
            dimension is the background.
        """
        pass
