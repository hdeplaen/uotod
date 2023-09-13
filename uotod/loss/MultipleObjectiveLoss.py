from typing import List
from warnings import warn

import torch
from torch import Tensor
import torch.nn.modules.loss as _Loss


# FIXME: list to *iter ?
# class MultipleObjectiveLoss(_Loss):  # FIXME: use _Loss instead iof module
class MultipleObjectiveLoss(torch.nn.Module):
    r"""
    Weighted sum of losses.
    """

    def __init__(self, losses: List = [], coefficients: List[float] = []):
        """
        :param losses: list of losses (with no reduction)
        :param coefficients: list of coefficients for the losses
        """
        super().__init__()

        assert len(losses) == len(coefficients), "The number of losses and weights must be the same."
        assert all([loss.reduction == "none" for loss in losses]), \
            "The reduction of the losses must be none."
        assert all([weight >= 0. for weight in coefficients]), "The coefficients must be non-negative."
        if len(losses) == 0:
            warn("The list of losses is empty, the loss will be set to zero.")

        for loss in losses:
            # check if "weight" is a parameter of the loss
            if hasattr(loss, "weight"):
                if len(losses) != 1:
                    raise NotImplementedError(f"The loss {loss} has a weight parameter, but there are several losses.")
                self.weight = loss.weight
                break

        self.losses = losses
        self.coefficients = coefficients

    def __repr__(self):
        return f"WeightedSumLoss(losses={self.losses}, weights={self.coefficients})"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: input tensor
        :param target: target tensor
        :return: weighted sum of the losses
        """
        assert input.shape[0] == target.shape[0], "The input and target tensors must have the same number of elements."

        loss = torch.zeros(input.shape[0], device=input.device, dtype=torch.float)
        for loss_, weight in zip(self.losses, self.coefficients):
            loss += weight * loss_(input, target)

        return loss