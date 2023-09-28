from typing import Optional
from warnings import warn
import re

import torch
from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Match)
class Min(_Match):
    r"""

    :param threshold: Threshold value. Defaults to 0.
    :param source: Either "target" (default) or "prediction".
    :type threshold: float, optional
    :type source: str, optional
    """

    @kwargs_decorator({"threshold": 0.0,
                       "source": "target",
                       "individual": False,
                       "unmatched_to_background": True})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(kwargs["threshold"])
        self.closest = kwargs["source"]
        self._unmatched_to_background = kwargs["unmatched_to_background"]

    @property
    def threshold(self) -> float:
        r"""
        Threshold value.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, val: float):
        val = float(val)
        self._threshold = val

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
        if val == "target" or val == 'targets':
            self._target_to_prediction = True
        elif val == "prediction" or val == 'predictions':
            self._target_to_prediction = False
        else:
            raise NameError(f"The value for the closest property can only be 'target' or 'prediction'.")

    @property
    def unmatched_to_background(self) -> bool:
        if not self._target_to_prediction:
            warn(
                "The unmatched_to_background property is irrelevant when not matching each target to the closest prediction")
        return self._unmatched_to_background

    @unmatched_to_background.setter
    def unmatched_to_background(self, val: bool):
        if not self._target_to_prediction:
            warn(
                "The unmatched_to_background property is irrelevant when not matching each target to the closest prediction")
        self._unmatched_to_background = val

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, target_mask: Optional[Tensor] = None,
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

        def assign_view(val: Tensor, with_background=False):
            nonlocal out_view
            if not with_background and self.background:
                out_view[:, :, :-1] = val
            else:
                out_view = val

        if self._target_to_prediction:  # each target is matched to the closest prediction
            vals, idx = get_cost(False).min(1, keepdim=True)
            idx_threshold = vals > self.threshold
            get_view(True).scatter_(1, idx, 1)  # fills in the minima
            assign_view(get_view(False).where(idx_threshold, 0), False)
            if target_mask is not None:  # suppresses the match to masked targets
                assign_view(get_view(False).where(target_mask.unsqueeze(1), 0), False)
            if self.background:
                if self._unmatched_to_background:  # unmatched predictions are matched towards the background
                    out_view[:, :, -1] = get_view(False).sum(dim=2).squeeze() == 0
                else:  # uniform match (corresponds to the limit case)
                    num_predictions = cost_matrix.size(1)
                    num_unmatched = num_predictions - get_view(False).sum(dim=2).sum(dim=1)
                    fill_value = num_unmatched / num_predictions
                    out_view[:, :, -1] = fill_value.unsqueeze(1)

        else:
            if target_mask is not None:
                assign_cost(get_cost(False).where(target_mask.unsqueeze(1), torch.inf), False)
            vals, idx = get_cost(True).min(2, keepdim=True)
            idx_threshold = vals > self.threshold
            get_view(True).scatter_(2, idx, 1)  # fills in the minima
            assign_view(get_view(True).where(idx_threshold, 0), True)
            if self.background and self._unmatched_to_background:  # unmatched predictions are matched towards the background
                out_view[:, :, -1] = get_view(False).sum(dim=2).squeeze() == 0

        return out_view
