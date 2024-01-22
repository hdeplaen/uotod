from typing import Optional
from warnings import warn

from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Match)
class ClosestPrediction(_Match):
    r"""
    Each target is matched to the closest prediction.

    :param threshold: Threshold value. Defaults to 0.
    :param uniform_background: Indicates whether the background should be uniform, which is the limit case of the UnbalancedOT (True), or
        only the unmatched predictions are matched to the background (False). Defaults to False.
    :type threshold: float, optional
    :type uniform_background: bool, optional
    """

    @kwargs_decorator({"threshold": 0.0,
                       "individual": False,
                       "uniform_background": False})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(kwargs["threshold"])
        self._uniform_background = kwargs["uniform_background"]

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
    def uniform_background(self) -> bool:
        if not self.background:
            warn(
                "The uniform_background property is irrelevant when no background is specified")
        return self._uniform_background

    @uniform_background.setter
    def uniform_background(self, val: bool):
        if not self.background:
            warn(
                "The uniform_background property is irrelevant when no background is specified")
        self._uniform_background = val

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

        vals, idx = get_cost(False).min(1, keepdim=True)
        idx_threshold = vals > self.threshold
        get_view(True).scatter_(1, idx, 1)  # fills in the minima
        assign_view(get_view(False).where(idx_threshold, 0), False)
        if target_mask is not None:  # suppresses the match to masked targets
            assign_view(get_view(False).where(target_mask.unsqueeze(1), 0), False)
        if self.background:
            if self._uniform_background:  # uniform match (corresponds to the limit case)
                num_predictions = cost_matrix.size(1)
                num_unmatched = num_predictions - get_view(False).sum(dim=2).sum(dim=1)
                fill_value = num_unmatched / num_predictions
                out_view[:, :, -1] = fill_value.unsqueeze(1)
            else:  # unmatched predictions are matched towards the background
                out_view[:, :, -1] = get_view(False).sum(dim=2).squeeze() == 0

        return out_view
