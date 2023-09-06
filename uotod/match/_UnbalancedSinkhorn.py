from typing import Optional, Tuple, Union
import sys
import importlib.util
import math
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ._Sinkhorn import _Sinkhorn
from ..utils import extend_docstring, kwargs_decorator


@extend_docstring(_Sinkhorn)
class _UnbalancedSinkhorn(_Sinkhorn, metaclass=ABCMeta):
    r"""
    :param ot_reg: Regularization parameter for the OT algorithm. Defaults to None.
    :type reg: float, optional
    """
    @kwargs_decorator({'reg': 0.1})
    def __init__(self,**kwargs) -> None:
        super(_UnbalancedSinkhorn, self).__init__(**kwargs)

        self.reg = kwargs['reg']

        assert isinstance(self.reg, float), \
            TypeError("The parameter reg must be a float.")

    @abstractmethod
    def _matching(self, hist_pred: Tensor, hist_tgt: Tensor, C:Tensor) -> Tensor:
        pass

    @extend_docstring(_Sinkhorn.compute_matching)
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
        hist_pred, hist_tgt = self._get_histograms(cost_matrix, target_mask)

        if self.normalize_cost_matrix:
            # Normalize the cost matrix to [0, 1]
            C = cost_matrix / torch.clamp(cost_matrix.max(), min=1e-8)
        else:
            C = cost_matrix

        # Compute the matching
        matching = self._matching(hist_pred, hist_tgt, C)
        matching = matching * num_pred

        return matching