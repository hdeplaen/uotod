from abc import ABCMeta, abstractmethod
from typing import Tuple

import math
import torch
from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Match)
class _Sinkhorn(_Match, metaclass=ABCMeta):
    r"""
    :param normalize_cost_matrix: Normalizes the cost matrix, defaults to True.
    :type normalize_cost_matrix: bool, optional
    :param reg_dimless: Dimensionless regularization parameter for the OT algorithm. Defaults to 0.12. This argument automatically sets the regularization `reg` depending on the problem size if the latter is not set. We refer to :cite:`De_Plaen_2023_CVPR` for more information.
    :param reg: Regularization parameter for the OT algorithm. Defaults to None. It is dependent of the problem size and we recommend leaving it blank and using the argument `reg_dimless` instead.
    :type reg_dimless: float, optional
    :type reg: float, optional
    """

    @kwargs_decorator({'normalize_cost_matrix': True,
                       'individual': False,
                       'reg_dimless': 0.12,
                       'reg': None})
    def __init__(self, **kwargs):
        super(_Sinkhorn, self).__init__(**kwargs)
        self.reg = kwargs['reg']
        self.reg_dimless = float(kwargs['reg_dimless'])

        assert isinstance(kwargs["normalize_cost_matrix"], bool), \
            TypeError("The argument normalize_cost_matrix must be a boolean.")
        self.normalize_cost_matrix = kwargs["normalize_cost_matrix"]

        if self.reg is not None:
            self.reg_dimless = None
            assert self.reg > 0., "The regularization parameter must be positive."
        else:
            assert self.reg_dimless is not None and self.reg_dimless > 0., "The adaptive regularization parameter must be positive."

    @torch.no_grad()
    def _get_histograms(self, cost_matrix: Tensor, target_mask: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        :param cost_matrix: cost matrix (batch_size, num_pred, num_tgt), padded with zeros
        :param target_mask: mask of the target boxes (batch_size, num_tgt)
        :return: histograms of the predictions and targets (batch_size, num_pred), (batch_size, num_tgt)
        """
        batch_size, num_pred, num_tgt = cost_matrix.shape
        num_tgt_batch = target_mask.sum(dim=1)  # number of target boxes in each batch element (batch_size)

        h_pred = torch.full((batch_size, num_pred), fill_value=1. / num_pred, device=cost_matrix.device,
                            requires_grad=False)
        h_tgt = torch.zeros((batch_size, num_tgt), device=cost_matrix.device, requires_grad=False)
        h_tgt[:, -1] = (num_pred - num_tgt_batch) / num_pred
        for b in range(batch_size):
            h_tgt[b, :num_tgt_batch[b]] = 1. / num_pred

        return h_pred, h_tgt

    @extend_docstring(_Match.compute_matching)
    @torch.no_grad()
    def compute_matching(self, cost_matrix: Tensor, target_mask: Tensor) -> Tensor:
        r"""
        Computes the matching between the predicted and target boxes. The optimal transport problem is solved
        using the Sinkhorn algorithm.
        :param cost_matrix: the cost matrix. Tensor of shape (batch_size, num_pred, num_tgt + 1).
        :param target_mask: the target mask. Tensor of shape (batch_size, num_tgt).
        :return: the matching. Tensor of shape (batch_size, num_pred, num_tgt + 1). The last entry of the last
            dimension is the background.
        """
        num_pred = cost_matrix.shape[1]
        # Compute the histograms
        hist_pred, hist_target = self._get_histograms(cost_matrix, target_mask)

        if self.normalize_cost_matrix:
            # Normalize the cost matrix to [0, 1]
            cost_matrix = cost_matrix / torch.clamp(cost_matrix.max(), min=1e-8)

        # Compute the regularization parameter
        if self.reg_dimless is not None:
            num_pred = cost_matrix.shape[1]
            reg = self.reg_dimless / (math.log(num_pred * 2.) + 1.)
        else:
            reg = self.reg

        matching = torch.zeros_like(cost_matrix)

        # Compute the matching
        if self._individual:
            for idx in range(cost_matrix.size(0)):
                matching[idx, :, :] = self._compute_matching_apart(cost_matrix.select(0, idx),
                                                                   matching.select(0, idx),
                                                                   reg=reg,
                                                                   hist_pred=hist_pred.select(0, idx),
                                                                   hist_target=hist_target.select(0, idx))
        else:
            matching = self._compute_matching_together(cost_matrix,
                                                       matching,
                                                       target_mask,
                                                       reg=reg,
                                                       hist_pred=hist_pred,
                                                       hist_target=hist_target)
        matching = matching * num_pred

        return matching
