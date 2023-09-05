import re

import torch
from torch import Tensor

from ._Match import _Match
from ..utils import kwargs_decorator, extend_docstring

@extend_docstring(_Match)
class Min(_Match):
    r"""

    :param threshold: Threshold value. Defaults to 0.
    :param target: Either "ground_truth" (default) or "prediction".
    :type threshold: float, optional
    :type target: str, optional
    """

    @kwargs_decorator({"threshold": 0.0,
                       "target": "ground_truth"})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(kwargs["threshold"])
        self.target = kwargs["threshold"]

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
    def target(self) -> str:
        if self._gt_target:
            return "ground_truth"
        else:
            return "prediction"

    @target.setter
    def target(self, val: str):
        assert isinstance(val, str), \
            TypeError("The property target is not set to a string.")
        val = re.sub(r'[^a-zA-Z]', '', val).lower()
        if val == "grounttruth":
            self._gt_target = True
        elif val == "prediction":
            self._gt_target = False
        else:
            raise NameError(f"The value for the target property can only be 'ground_truth' or 'prediction'.")

    def compute_matching(self, cost_matrix: Tensor, target_mask: Tensor) -> Tensor:
        pass