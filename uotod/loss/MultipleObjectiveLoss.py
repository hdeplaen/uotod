from typing import List
from warnings import warn

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MultipleObjectiveLoss(_Loss):
    r"""
    Creates a criterion that combines several losses with different weights.

    .. math::
        \text{loss} = \sum_{i=1}^n w_i \text{loss}_i

    :param losses: list of losses, with reduction="none" for all losses
    :type losses: List[_Loss]
    :param weights: list of weights for the losses
    :type weights: List[float]
    :return: weighted sum of the losses
    :rtype: Tensor (float)

    Example:
        >>> import torch
        >>> from uotod.loss import MultipleObjectiveLoss, IoULoss
        >>> loss = MultipleObjectiveLoss(
        >>>     [IoULoss(reduction="none"), torch.nn.L1Loss(reduction="none")],
        >>>     [1., 2.]
        >>> )
    """

    def __init__(self, losses: List[_Loss] = None, weights: List[float] = None) -> None:
        super().__init__()

        assert len(losses) == len(weights), "The number of losses and weights must be the same."
        assert all([loss.reduction == "none" for loss in losses]), \
            "The reduction of the losses must be none."
        assert all([weight >= 0. for weight in weights]), "The coefficients must be non-negative."

        for loss in losses:
            # check if "weight" is a parameter of the loss (for classification losses)
            if hasattr(loss, "weight"):
                if len(losses) != 1:
                    raise NotImplementedError(f"The loss {loss} has a weight parameter, but there are several losses.")
                self.weight = loss.weight
                break

        self.losses = losses
        self.coefficients = weights
        self.reduction = "none"  # the reduction is always "none" for this loss

    def __repr__(self):
        return f"MultipleObjectiveLoss(losses={self.losses}, weights={self.coefficients})"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        :param input: input tensor
        :param target: target tensor
        :type input: Tensor of shape (batch_size, ...)
        :type target: Tensor of shape (batch_size, ...)
        :return: weighted sum of the losses
        :rtype: Tensor (float)
        """
        assert input.shape[0] == target.shape[0], "The input and target tensors must have the same number of elements."

        loss = torch.zeros(input.shape[0], device=input.device, dtype=torch.float)
        for loss_fn, w in zip(self.losses, self.coefficients):
            loss_ = loss_fn(input, target)
            # sum over all dimensions except the batch dimension
            if loss_.dim() > 1:
                loss_ = loss_.sum(dim=tuple(range(1, loss_.dim())))
            loss += w * loss_

        return loss
