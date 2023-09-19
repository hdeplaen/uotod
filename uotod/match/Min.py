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
                                **kwargs) -> None:
        if self._target_to_prediction:  # each target is matched to the closest prediction
            out_view.scatter_(0, cost_matrix[:, :-1].argmin(0, keepdim=True), 1)  # fills in the minima
            if target_mask is not None:  # suppresses the match to masked targets
                out_view[:, :-1] = torch.where(target_mask, out_view[:, :-1], 0)
            if self._unmatched_to_background:  # unmatched predictions are matched towards the background
                out_view[:, -1] = out_view[:, :-1].sum(dim=1).squeeze() == 0
            else:  # uniform match (corresponds to the limit case)
                num_predictions = cost_matrix.size(dim=0)
                if target_mask is not None:
                    num_targets = target_mask.sum(0)
                    fill_value = (num_predictions - num_targets) / num_predictions
                else:
                    num_targets = cost_matrix.size(dim=1)
                    fill_value = ((num_predictions - num_targets) / num_predictions)
                out_view[:, -1] = fill_value
        else:
            if target_mask is not None:
                torch.where(target_mask, cost_matrix[:, :-1], torch.inf)
            out_view.scatter_(1, cost_matrix.argmin(1, keepdim=True), 1)

    def _compute_matching_together(self, cost_matrix: Tensor, out_view: Tensor, target_mask: Optional[Tensor] = None,
                                   **kwargs) -> None:
        if self.background:
            cost_without_background = cost_matrix[:, :, :-1]
            out_view_without_background = out_view[:, :, :-1]
        else:
            cost_without_background = cost_matrix[:, :, :]
            out_view_without_background = out_view[:, :, :]

        if self._target_to_prediction:  # each target is matched to the closest prediction
            vals, idx = cost_without_background.min(1, keepdim=True)
            idx_threshold = vals < self.threshold
            out_view.scatter_(1, idx, 1)  # fills in the minima
            out_view_without_background = torch.where(idx_threshold, out_view_without_background, 0)
            if target_mask is not None:  # suppresses the match to masked targets
                out_view_without_background = torch.where(target_mask, out_view_without_background, 0)
            if self.background:
                if self._unmatched_to_background:  # unmatched predictions are matched towards the background
                    out_view_without_background = out_view_without_background.sum(dim=2).squeeze() == 0
                else:  # uniform match (corresponds to the limit case)
                    num_predictions = cost_matrix.size(dim=1)
                    if target_mask is not None:
                        num_targets = target_mask.sum(1)
                        fill_value = (num_predictions - num_targets) / num_predictions
                    else:
                        num_targets = cost_matrix.size(dim=2)
                        num_batch = cost_matrix.size(dim=0)
                        fill_value = ((num_predictions - num_targets) / num_predictions).expand(num_batch)
                    out_view_without_background = fill_value
        else:
            if target_mask is not None:
                cost_without_background = torch.where(target_mask, cost_without_background, torch.inf)
            out_view.scatter_(2, cost_matrix.argmin(2, keepdim=True), 1)
