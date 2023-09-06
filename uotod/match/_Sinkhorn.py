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
    """
    @kwargs_decorator({'normalize_cost_matrix': True})
    def __init__(self, **kwargs):
        super(_Sinkhorn, self).__init__(**kwargs)
        assert isinstance(kwargs["normalize_cost_matrix"], bool), \
            TypeError("The argument normalize_cost_matrix must be a boolean.")
        self.normalize_cost_matrix = kwargs["normalize_cost_matrix"]

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