from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from scipy.optimize import linear_sum_assignment

from ._Match import _Match
from ..utils import extend_docstring, kwargs_decorator


@extend_docstring(_Match)
class Hungarian(_Match):
    r"""
    :param scipy: Uses SciPy's implementation if `True` (default). Otherwise, uses the implementation of `this repo <https://github.com/bkj/auction-lap>`_, following from :cite:`bertsekas1992auction`.
    :type scipy: bool, optional
    """

    @kwargs_decorator({'scipy': True})
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._scipy = kwargs['scipy']
        assert isinstance(self._scipy, bool), "The argument scipy has to be a boolean."

    @torch.no_grad()
    def _bkj_auction(self, cost_matrix, eps=None) -> Tensor:
        """
        cost_matrix: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime

        SOURCE: https://github.com/bkj/auction-lap
        """
        raise NotImplementedError("Coming soon...")

        eps = 1 / cost_matrix.shape[0] if eps is None else eps

        # --
        # Init

        cost = torch.zeros((1, cost_matrix.shape[1]))
        curr_ass = torch.zeros(cost_matrix.shape[0]).long() - 1
        bids = torch.zeros(cost_matrix.shape)

        if cost_matrix.is_cuda:
            cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()

        while (curr_ass == -1).any():
            # --
            # Bidding

            unassigned = (curr_ass == -1).nonzero().squeeze()

            value = cost_matrix[unassigned] - cost
            top_value, top_idx = value.topk(2, dim=1)

            first_idx = top_idx[:, 0]
            first_value, second_value = top_value[:, 0], top_value[:, 1]

            bid_increments = first_value - second_value + eps

            bids_ = bids[unassigned]
            bids_.zero_()
            bids_.scatter_(
                dim=1,
                index=first_idx.contiguous().view(-1, 1),
                src=bid_increments.view(-1, 1)
            )

            # --
            # Assignment

            have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

            high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
            high_bidders = unassigned[high_bidders.squeeze()]

            cost[:, have_bidder] += high_bids

            curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
            curr_ass[high_bidders] = have_bidder.squeeze()

        return curr_ass

    @torch.no_grad()
    def _scipy_auction(self, cost_matrix) -> Tensor:
        device = cost_matrix.device
        # Move to CPU
        cost_matrix = cost_matrix.cpu()

        # Compute the matching
        matching = torch.zeros_like(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        matching[row_ind, col_ind] = 1.

        # Move back to GPU
        matching = matching.to(device)

        return matching

    @torch.no_grad()
    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor) -> Tensor:
        r"""
        This method makes use of Scipy's `linear_sum_assignment`.
        """

        if self.background:
            if self._scipy:
                out_view[:, :-1] = self._scipy_auction(cost_matrix[:, :-1])
            else:
                out_view[:, :-1] = self._bkj_auction(cost_matrix[:, :-1])

            # Assign unmatched predictions to the background
            out_view[:, -1] = 1. - out_view[:, :-1].sum(dim=1)
        else:
            if self._scipy:
                out_view = self._scipy_auction(cost_matrix)
            else:
                out_view = self._bkj_auction(cost_matrix)
        return out_view
