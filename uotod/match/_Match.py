from abc import ABCMeta, abstractmethod
from typing import Dict, Union, List, Optional, Tuple
from warnings import warn
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ..utils import convert_target_to_dict, kwargs_decorator
from ..plot import image_with_boxes, match, cost


class _Match(nn.Module, metaclass=ABCMeta):
    r"""
    :param cls_match_module: Classification loss used to compute the matching, if any.
    :param loc_match_module: Localization loss used to compute the matching, if any.
    :param background: Indicates whether there is a background. Defaults to True.
    :param background_cost: Cost of the background class. Defaults to 10.
    :param is_anchor_based: If True, the matching is performed between the anchor boxes and the target boxes.
    :type cls_match_module: _Loss
    :type loc_match_module: _Loss
    :type background: bool, optional
    :type background_cost: float, optional
    :type is_anchor_based: bool, optional
    """

    @kwargs_decorator({'cls_match_module': None,
                       'loc_match_module': None,
                       'background': True,
                       'background_cost': 10.0,
                       'is_anchor_based': False,
                       'individual': True})
    def __init__(self, **kwargs) -> None:
        super(_Match, self).__init__()

        assert isinstance(kwargs['background'], bool), 'The background argument must be set to either True or False.'
        assert kwargs['background_cost'] >= 0., "The background cost must non-negative."

        self.matching_cls_module = kwargs['cls_match_module']
        self.matching_loc_module = kwargs['loc_match_module']

        if self.matching_cls_module is not None:
            if self.matching_cls_module.reduction != 'none': raise ValueError("The classification cost module must have a reduction set to 'none'.")
        if self.matching_loc_module is not None:
            if self.matching_loc_module.reduction != 'none': raise ValueError("The localization cost module must have a reduction set to 'none'.")

        self.background = kwargs['background']
        self.background_cost = kwargs['background_cost']
        self.is_anchor_based = kwargs['is_anchor_based']

        self._individual = kwargs['individual']

    @torch.no_grad()
    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None,
                cost_matrix: Optional[Tensor] = None,
                save: bool= True) \
            -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Computes a batch of matchings between the predicted and target boxes.

        :param input: Input containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type input: dictionary
        :param target: Target containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :type target: dictionary
        :param anchors: the anchors used to compute the predicted boxes.
            (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :return: the matching between the predicted and target boxes, and the cost matrix if returns_cost is True:
            Tensor of shape (batch_size, num_pred, num_targets + 1). The last entry of the last dimension is the
            background.
        :rtype: Tensor (float) or Tuple(Tensor, Tensor)
        """
        if target["mask"] is not None:
            assert target["mask"].dtype == torch.bool, "The target mask must be of type bool."

        # emptying the cache
        self._last_cost = None
        self._last_match = None
        self._last_input = None
        self._last_target = None

        # Compute the cost matrix
        if cost_matrix is None:
            cost_matrix = self.compute_cost_matrix(input, target, anchors)

        # Compute the matching.
        matching = self.compute_matching(cost_matrix, target["mask"])

        # Verify that the matching is not NaN.
        if torch.isnan(matching).any():
            raise RuntimeError("The matching is NaN, this may be caused by a too low regularization parameter.")

        # saving for the plotting
        if save:
            self._last_cost = cost_matrix
            self._last_match = matching
            self._last_input = input
            self._last_target = target

        return matching

    def compute_cost_matrix(self,
                            input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                            target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                            anchors: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes a batch of cost matrices between the predicted and target boxes.

        :param input: Input containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type input: dictionary
        :param target: Target containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :type target: dictionary
        :param anchors: the anchors used to compute the predicted boxes.
            (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :param background: Indicated whether the background has to be added.
        :type background: bool, optional
        :return: the matching between the predicted and target boxes:
            Tensor of shape (batch_size, num_pred, num_targets + 1) or (batch_size, num_pred, num_targets) if
            background is False.
        :rtype: Tensor (float)
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
            raise ValueError("Both localization and classification costs are undefined.")

        # Mask the cost matrix.
        cost_matrix = cost_matrix * target["mask"].unsqueeze(dim=1).expand(cost_matrix.shape)

        # Add the background cost.
        if self.background:
            cost_matrix = torch.cat([cost_matrix, self.background_cost * torch.ones_like(cost_matrix[:, :, :1])],
                                    dim=2)

        return cost_matrix

    def _compute_cls_costs(self, pred_logits: Tensor, tgt_classes: Tensor) -> Optional[Tensor]:
        r"""
        Computes the classification cost matrix.
        :param pred_logits: Predicted logits. Tensor of shape (batch_size, num_pred, num_classes).
        :param tgt_classes: Target classes. Tensor of shape (batch_size, num_targets).
        :return: the classification cost matrix. Tensor of shape (batch_size, num_pred, num_classes).
        """
        if self.matching_loc_module is None and self.matching_cls_module is None:
            raise ValueError("At least a classification or a localization loss module must be provided.")

        if self.matching_cls_module is None:
            return None

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
        loc_cost = self.matching_loc_module(pred_locs_rep, tgt_locs_rep)
        if loc_cost.dim() == 2:
            loc_cost = loc_cost.sum(dim=1)
        loc_cost = loc_cost.view(batch_size, num_pred, num_tgt)

        return loc_cost

    def compute_matching(self, cost_matrix: Tensor, target_mask: Optional[Tensor]) -> Tensor:
        r"""
        Computes the matching.

        :param cost_matrix: Cost matrix of shape (batch_size, num_pred, num_targets + 1).
        :param target_mask: Target mask of shape (batch_size, num_targets).
        :type cost_matrix: Tensor
        :type target_mask: BoolTensor, optional
        :return: The matching :math:`\mathbf{P}` for each element of the batch. Tensor of shape (batch_size, num_pred, num_targets + 1). The last entry of the last
            dimension [:, :, num_target+1] is the background.
        """
        p = torch.zeros_like(cost_matrix)
        if target_mask is None:
            if self._individual:
                for idx in range(cost_matrix.size(0)):
                    p[idx, :, :] = self._compute_matching_apart(cost_matrix.select(0, idx),
                                                                p.select(0, idx))
            else:
                p = self._compute_matching_together(cost_matrix, p)
        else:
            if self._individual:
                for idx in range(cost_matrix.size(0)):
                    target_mask_batch = torch.cat([target_mask[idx, :], torch.ones(1, dtype=torch.bool,
                                                                                   device=target_mask.device)])
                    p[idx, :, target_mask_batch] = self._compute_matching_apart(
                        cost_matrix.select(0, idx)[:, target_mask_batch],
                        p.select(0, idx)[:, target_mask_batch]
                    )
            else:
                p = self._compute_matching_together(cost_matrix, p, target_mask)
        return p

    def _compute_matching_together(self, cost_matrix: Tensor, out_view: Tensor, target_mask: Optional[Tensor] = None,
                                   **kwargs) -> Tensor:
        raise Exception('Parallelism is not possible for this matching. Please compute each match individually by '
                        'not setting the argument individual to True during construction.')

    @abstractmethod
    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, target_mask: Optional[Tensor] = None,
                                **kwargs) -> Tensor:
        raise NotImplementedError

    def plot(self, idx=0, img: Optional[Union[Tensor, ndarray]] = None,
             plot_cost: bool = True,
             plot_match: bool = True,
             max_background_match: Union[float, int] = 1.,
             background: bool = True,
             erase: bool = False):
        r"""
        Plots from the last batch
        # TODO: extensive description

        :param idx: Index of the image to be plotted.
        :type idx: int, optional
        :param img: Image to be plotted. If it is not None, the boxes plot is computed.
        :type img: Tensor or ndarray, optional
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
        exception = Exception(
            'The model has saved no values to plot. Insure that forward has been called at least once with the save argument set to True (by default).')

        assert self._last_match is not None, exception
        assert self._last_target is not None, exception
        assert self._last_cost is not None, exception
        assert self._last_input is not None, exception

        if not self.background:
            warn('Cannot plot the background class as the model does not contain one')
            background = False

        matching_matrix = self._last_match[idx, :, :]
        pred_mask = matching_matrix[:, -1] <= max_background_match
        target_mask = self._last_target["mask"][idx, :]

        # IMAGE WITH BOXES
        if img is not None:
            if isinstance(img, ndarray) or isinstance(img, Tensor):
                img_loc = img
            elif isinstance(img, list):
                img_loc = img[idx]
            else:
                raise Exception("The img argument must be an ndarray or Tensor for one image or a list for multiple.")
            fig_img = image_with_boxes(img_loc,
                                       self._last_input["pred_boxes"][idx, :, :],
                                       self._last_target["boxes"][idx, :, :],
                                       pred_mask, target_mask)
        else:
            fig_img = None

        # COST
        if plot_cost:
            fig_cost = cost(self._last_cost[idx, :, :],
                            pred_mask, target_mask, background=background)
        else:
            fig_cost = None

        # MATCH
        if plot_match:
            fig_match = match(matching_matrix, pred_mask, target_mask, background=background)
        else:
            fig_match = None

        if erase:
            self._last_cost = None
            self._last_match = None
            self._last_input = None
            self._last_target = None


        return fig_img, fig_cost, fig_match
