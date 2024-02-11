from typing import Optional
from warnings import warn
import re

import torch
from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Match)
class SoftMin(_Match):
    r"""
    :param source: Either "target" (default) or "prediction".
    :type source: str, optional
    """

    @kwargs_decorator({"source": "target",
                       "individual": False,
                       "background_weight": True,
                       "reg":1.})
    def __init__(self, **kwargs):
        super(SoftMin, self).__init__(**kwargs)
        self.closest = kwargs["source"]
        self.background_weight = kwargs["background_weight"]
        self.reg = kwargs["reg"]
        assert self.reg > 0, 'The regularization parameter has to be strictly greater than 0.'


    @property
    def closest(self) -> str:
        if self._target_to_prediction:
            return "target"
        else:
            return "prediction"

    @closest.setter
    def closest(self, val: str):
        assert isinstance(val, str), \
            TypeError("The property target is not set to a string.")
        val = re.sub(r'[^a-zA-Z]', '', val).lower()
        if val == "target" or val == "targets":
            self._target_to_prediction = True
        elif val == "prediction" or val == "predictions":
            self._target_to_prediction = False
        else:
            raise NameError(f"The value for the closest can only be 'target' or 'prediction'.")

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, **kwargs) -> Tensor:
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

        def assign_view(val: Tensor, with_background=False):
            nonlocal out_view
            if not with_background and self.background:
                out_view[:, :, :-1] = val
            else:
                out_view = val

        if self.reg != 1.:
            inv_reg = 1/self.reg
            cost_matrix_reg = inv_reg * cost_matrix
        else:
            cost_matrix_reg = cost_matrix

        if self._target_to_prediction:  # each target is matched to the closest prediction
            out_view = torch.nn.functional.softmax(-cost_matrix_reg, dim=1)
            if target_mask is not None:  # suppresses the match to masked targets
                assign_view(get_view(False).where(target_mask.unsqueeze(1), 0), False)
            if self.background and self.background_weight:  # put a higher weight on the background
                num_predictions = cost_matrix_reg.size(dim=1)
                if target_mask is None:
                    num_targets = cost_matrix_reg.size(dim=2)
                else:
                    num_targets = target_mask.sum(1, keepdim=True)
                out_view[:, :, -1] = out_view[:, :, -1] * (num_predictions - num_targets)

        else:
            if target_mask is not None:
                assign_cost(get_cost(False).where(target_mask.unsqueeze(1), torch.inf), False)
            out_view = torch.nn.functional.softmax(-cost_matrix_reg, dim=2)

        return out_view
