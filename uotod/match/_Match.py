from abc import ABCMeta, abstractmethod
from typing import Dict, Union, List, Optional, Tuple
from warnings import warn
from numpy import array

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ..utils import convert_target_to_dict, kwargs_decorator
from ..plot import image_with_boxes, match, cost

class _Match(nn.Module, metaclass=ABCMeta):
    r"""
    :param cls_matching_module: Classification loss used to compute the matching, if any.
    :param loc_matching_module: Localization loss used to compute the matching, if any.
    :param bg_class_position: Index of the background class. "first", "last" or "none" (no background class).
    :param bg_cost: Cost of the background class.
    :param is_anchor_based: If True, the matching is performed between the anchor boxes and the target boxes.
    :type bg_class_position: str, optional
    :type bg_cost: float, optional
    :type is_anchor_based: bool, optional
    """

    @kwargs_decorator({'cls_matching_module': None,
                       'loc_matching_module': None,
                       'bg_class_position': "first",
                       'bg_cost': 10.0,
                       'is_anchor_based': False})
    def __init__(self, **kwargs) -> None:
        super().__init__()

        assert isinstance(kwargs['cls_matching_module'], _Loss) or isinstance(kwargs['loc_matching_module'], _Loss), \
            "At least one of the classification and localization losses must be provided."
        assert kwargs['bg_cost'] >= 0., "The background cost must a non-negative float."
        assert kwargs['bg_class_position'] in ["first", "last", "none"], \
            "bg_class_index must be 'first', 'last' or 'none'"

        self.matching_cls_module = kwargs['cls_matching_module']
        self.matching_loc_module = kwargs['loc_matching_module']

        assert self.matching_cls_module is not None or self.matching_loc_module is not None, \
            "At least a localization or classification cost must be provided, but both are None."

        self.bg_class_position = kwargs['bg_class_position']  # FIXME: this is not used; but do we need it?
        self.bg_cost = kwargs['bg_cost']
        self.is_anchor_based = kwargs['is_anchor_based']

    @torch.no_grad()
    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None, returns_cost: bool = False) \
            -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Computes a batch of matchings between the predicted and target boxes.

        :param input: Input containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type input: dictionnary
        :param target: Target containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :type target: dictionnary
        :param anchors: the anchors used to compute the predicted boxes.
            (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :return: the matching between the predicted and target boxes:
            Tensor of shape (batch_size, num_pred, num_targets).
        :rtype: Tensor

        """
        # emptying the cache
        self._last_cost = None
        self._last_match = None
        self._last_input = None
        self._last_target = None

        # Compute the cost matrix
        cost_matrix = self.compute_cost_matrix(input, target, anchors)

        # Compute the matching.
        matching = self.compute_matching(cost_matrix, target["mask"])

        # Verify that the matching is not NaN.
        if torch.isnan(matching).any():
            raise RuntimeError("The matching is NaN, this may be caused by a too low regularization parameter.")

        # saving for the plotting
        self._last_cost = cost_matrix
        self._last_match = matching
        self._last_input = input
        self._last_target = target

        if returns_cost:
            return matching, cost_matrix
        return matching

    def compute_cost_matrix(self,
                            input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                            target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                            anchors: Optional[Tensor] = None, background: bool = True) -> Tensor:
        r"""
        Computes a batch of cost matrices between the predicted and target boxes.

        :param input: Input containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type input: dictionnary
        :param target: Target containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :type target: dictionnary
        :param anchors: the anchors used to compute the predicted boxes.
            (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :param background: Indicated whether the background has to be added.
        :type background: bool, optional
        :return: the matching between the predicted and target boxes:
            Tensor of shape (batch_size, num_pred, num_targets).
        :rtype: Tensor
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

        if cls_cost is not None and loc_cost is not None:
            cost_matrix = loc_cost + cls_cost
        elif cls_cost is None:
            cost_matrix = loc_cost
        elif loc_cost is None:
            cost_matrix = cls_cost
        else:
            raise Exception("Both localization and classification costs are undefined.")

        # Mask the cost matrix.
        cost_matrix = cost_matrix * target["mask"].unsqueeze(dim=1).expand(cost_matrix.shape)

        # Add the background cost.
        if background: cost_matrix = torch.cat([cost_matrix, self.bg_cost * torch.ones_like(cost_matrix[:, :, :1])], dim=2)

        return cost_matrix

    def _compute_cls_costs(self, pred_logits: Tensor, tgt_classes: Tensor) -> Optional[Tensor]:
        r"""
        Computes the classification cost matrix.
        :param pred_logits: Predicted logits. Tensor of shape (batch_size, num_pred, num_classes).
        :param tgt_classes: Target classes. Tensor of shape (batch_size, num_targets).
        :return: the classification cost matrix. Tensor of shape (batch_size, num_pred, num_classes).
        """
        if self.matching_cls_module is None:
            return None

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

    def _compute_loc_costs(self, pred_locs: Tensor, tgt_locs: Tensor) -> Optional[Tensor]:
        r"""
        Computes the localization cost matrix.

        :param pred_locs: Predicted locations. Tensor of shape (batch_size, num_pred, 4),
        :param tgt_locs: Target locations. Tensor of shape (batch_size, num_targets, 4),
        :return: the localization cost matrix. Tensor of shape (batch_size, num_pred, num_targets).
        """
        if self.matching_loc_module is None:
            return None

        # TODO: avoid redundancy of same costs are used.
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

    def plot(self, idx=0, img: Optional[Union[Tensor, array]] = None,
             plot_cost: bool = True,
             plot_match: bool = True,
             max_background_match: float = 1.,
             background: bool = True):
        r"""
        Plots from the last batch

        :param idx: Index of the image to be plotted.
        :type idx: int, optional
        :param img: Image to be plotted. If it is not None, the boxes plot is computed.
        :type img: Tensor or array, optional
        :param plot_cost: Plots the cost matrix between the predictions and the targets, including background.
        :type plot_cost: bool, optional
        :param plot_match: Plots the cost matrix between the predictions and the targets, including background.
        :type plot_match: bool, optional
        :param max_background_match: A threshold to only plot relevant matched predictions.
            The predictions are only plotted if the value matched to the background does not exceed
            `max_background_match`. Defaults to 1.
        :type max_background_match: float, optional
        :return: Matplotlib figures
        :rtype: Tuple(fig, fig, fig)
        """

        matching_matrix = self._last_match[idx,:,:]
        pred_mask = matching_matrix[:,-1] <= max_background_match
        target_mask = self._last_target["mask"][idx,:]

        # IMAGE WITH BOXES
        if img is not None:
            fig_img = image_with_boxes(img,
                                       self._last_input["pred_boxes"][idx,:,:],
                                       self._last_target["boxes"][idx,:,:],
                                       pred_mask, target_mask)
        else: fig_img = None

        # COST
        if plot_cost:
            fig_cost = cost(self._last_cost[idx,:,:],
                            pred_mask, target_mask, background=background)
        else: fig_cost = None

        # MATCH
        if plot_match:
            fig_match = match(matching_matrix, pred_mask, target_mask, background=background)
        else: fig_match = None

        return fig_img, fig_cost, fig_match











































