from typing import Optional
from warnings import warn
import re

import torch
from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Match)
class ClosestTarget(_Match):
    r"""
    Each prediction is matched to the closest target.
    """

    @kwargs_decorator({"individual": False})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor,
                                **kwargs) -> Tensor:
        raise NotImplementedError

    def _compute_matching_together(self, cost_matrix: Tensor, out_view: Tensor, target_mask: Optional[Tensor] = None,
                                   **kwargs) -> Tensor:
        def get_cost(with_background=False) -> Tensor:
            if not with_background and self.background:
                return cost_matrix[:, :, :-1]
            return cost_matrix

        def get_view(with_background=False) -> Tensor:
            if not with_background and self.background:
                return out_view[:, :, :-1]
            return out_view

        def assign_cost(val: Tensor, with_background=False):
            nonlocal cost_matrix
            if not with_background and self.background:
                cost_matrix[:, :, :-1] = val
            else:
                cost_matrix = val

        if target_mask is not None:
            assign_cost(get_cost(False).where(target_mask.unsqueeze(1), torch.inf), False)
        _, idx = get_cost(True).min(2, keepdim=True)
        get_view(True).scatter_(2, idx, 1)  # fills in the minima

        return out_view
