from typing import Dict, Optional, Union, Tuple
from warnings import warn
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ._MatchingMethod import _MatchingMethod
from ..loss.modules import MultipleObjectiveLoss

import sys
import importlib.util



class _MatchingMethod(nn.Module):
    pass


class _SinkhornBalancedNative(_MatchingMethod):
    @torch.no_grad()
    def forward(self, hist_pred: Tensor, hist_tgt: Tensor, cost_matrix: Tensor, reg: float) -> Tensor:
        """
        :param hist_pred: histogram of the predictions (batch_size, num_pred)
        :param hist_tgt: histogram of the targets (batch_size, num_tgt)
        :param cost_matrix: cost matrix (batch_size, num_pred, num_tgt), padded with zerosµ
        :param reg: regularization parameter
        :return: coupling matrix (batch_size, num_pred, num_tgt)
        """
        # Preliminaries
        batch_size, num_pred, _ = cost_matrix.shape

        # Initialization of the algorithm
        K = torch.exp(-cost_matrix / reg)
        u = torch.ones_like(hist_pred, requires_grad=False)

        # Iterations
        for _ in range(self.num_iter):
            u = hist_pred / (K * (hist_tgt / (K * u.unsqueeze(2)).sum(dim=1)).unsqueeze(1)).sum(dim=2)

        # Coupling matrix P = diag(u) @ K @ diag(v)
        P = torch.einsum("ni,nij,nj->nij", u, K, hist_tgt / (K * u.unsqueeze(2)).sum(dim=1))

        return P.data

class _SinkhornBalancedEpsilon(_MatchingMethod):
    def __init__(self, num_iter: int):
        """
        Sinkhorn algorithm for balanced OT
        :param num_iter: number of iterations
        """
        super().__init__()
        self.method = _check_ot_installed().bregman.sinkhorn_epsilon_scaling

    def forward(self, hist_pred:Tensor,
                hist_tgt:Tensor,
                cost_matrix:Tensor,
                reg:float,
                **kwargs):
        return self.method(a=hist_pred,
                           b=hist_tgt,
                           M=cost_matrix,
                           reg=reg,
                           numItermax=self.num_iter,
                           **kwargs)[0]


class SinkhornUnbalanced(nn.Module):
    def __init__(self, num_iter: int, tau1: float, tau2: float):
        """
        Sinkhorn algorithm for unbalanced OT
        :param num_iter: number of iterations
        :param tau1: regularization parameter for the source histogram, ignored if None
        :param tau2: regularization parameter for the target histogram, ignored if None
        """
        super().__init__()

        self.num_iter = num_iter
        self.tau1 = tau1
        self.tau2 = tau2

    @torch.no_grad()
    def forward(self, hist_pred: Tensor, hist_tgt: Tensor, cost_matrix: Tensor, reg: float) -> Tensor:
        """
        :param hist_pred: histogram of the predictions (batch_size, num_pred)
        :param hist_tgt: histogram of the targets (batch_size, num_tgt)
        :param cost_matrix: cost matrix (batch_size, num_pred, num_tgt), padded with zeros
        :param reg: regularization parameter
        :return: coupling matrix (batch_size, num_pred, num_tgt)
        """
        # Preliminaries
        batch_size, num_pred, _ = cost_matrix.shape
        factor1 = self.tau1 / (self.tau1 + reg) if self.tau1 is not None else 1.
        factor2 = self.tau2 / (self.tau2 + reg) if self.tau2 is not None else 1.

        # Initialization of the algorithm
        K = torch.exp(-cost_matrix / reg)
        u = torch.ones_like(hist_pred, requires_grad=False)

        # Iterations
        for _ in range(self.num_iter):
            u = (hist_pred / (K * ((hist_tgt / (K * u.unsqueeze(2)).sum(dim=1)).pow(factor2))
                              .unsqueeze(1)).sum(dim=2)).pow(factor1)

        # Coupling matrix P = diag(u) @ K @ diag(v)
        P = torch.einsum("ni,nij,nj->nij", u, K, (hist_tgt / (K * u.unsqueeze(2)).sum(dim=1)).pow(factor2))

        return P.data


class OTMatching(_MatchingMethod):
    def __init__(self,
                 cls_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 loc_matching_module: Union[MultipleObjectiveLoss, _Loss, None] = None,
                 bg_class_position: str = "first",
                 bg_cost: float = 10.0,
                 is_anchor_based: bool = False,
                 ot_num_iter: int = 50,
                 ot_reg0: Union[float, None] = 0.12,
                 ot_reg: Union[float, None] = None,
                 ot_tau1: Union[float, None] = None,
                 ot_tau2: Union[float, None] = None,
                 normalize_cost_matrix: bool = True) -> None:
        r"""
        OT Matching. The matching is performed by solving an optimal transport problem between the predicted and
        target histograms. The cost matrix is computed as the sum of the classification and localization losses. The
        optimal transport problem is solved using the Sinkhorn algorithm.
        :param cls_matching_module: Classification loss used to compute the cost matrix.
        :param loc_matching_module: Localization loss used to compute the cost matrix.
        :param bg_class_position: Position of the background class in the classification output. Can be "first" or "last".
        :param bg_cost: Cost of the background class.
        :param is_anchor_based: If True, the matching is performed between the anchor boxes and the target boxes.
        :param ot_num_iter: Number of iterations for the OT algorithm.
        :param ot_reg0: Adaptive regularization parameter for the OT algorithm.
        :param ot_reg: Regularization parameter for the OT algorithm.
        :param ot_tau1: First constraint parameter for the unbalanced OT algorithm (if None, the constraint is enforced).
        :param ot_tau2: Second constraint parameter for the unbalanced OT algorithm (if None, the constraint is enforced).
        :param normalize_cost_matrix: If True, the cost matrix is normalized to [0, 1].
        """
        super().__init__(cls_matching_module, loc_matching_module, bg_class_position, bg_cost, is_anchor_based)

        assert ot_num_iter > 0, "The number of iterations must be positive."
        if ot_reg is not None:
            ot_reg0 = None
            assert ot_reg > 0., "The regularization parameter must be positive."
        else:
            assert ot_reg0 is not None and ot_reg0 > 0., "The adaptive regularization parameter must be positive."
        assert ot_tau1 is None or ot_tau1 >= 0., "The first constraint parameter must be non-negative."
        assert ot_tau2 is None or ot_tau2 >= 0., "The second constraint parameter must be non-negative."

        if ot_tau1 is None and ot_tau2 is None:
            warn("Running the balanced OT algorithm.")
        """elif ot_tau1 == 0. and ot_tau2 is None:
            warn("Matching each prediction to the closest target.")
        elif ot_tau1 is None and ot_tau2 == 0.:
            warn("Matching each target to the closest prediction.")"""

        self.ot_num_iter = ot_num_iter
        self.ot_reg0 = ot_reg0
        self.ot_reg = ot_reg
        self.ot_tau1 = ot_tau1
        self.ot_tau2 = ot_tau2
        self.normalize_cost_matrix = normalize_cost_matrix

        # TODO: add option to use CUDA implementation
        if ot_tau1 is None and ot_tau2 is None:
            self.matching_fn = SinkhornBalanced(num_iter=ot_num_iter)
        else:
            self.matching_fn = SinkhornUnbalanced(num_iter=ot_num_iter, tau1=ot_tau1, tau2=ot_tau2)



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

        # Compute the regularization parameter
        if self.ot_reg0 is not None:
            num_pred = cost_matrix.shape[1]
            reg = self.ot_reg0 / (math.log(num_pred * 2.) + 1.)
        else:
            reg = self.ot_reg

        # Compute the matching
        matching = self.matching_fn(hist_pred, hist_tgt, C, reg)
        matching = matching * num_pred

        return matching
