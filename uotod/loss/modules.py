from typing import List
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchvision.ops import generalized_box_iou_loss, distance_box_iou_loss


# FIXME: list to *iter ?
class MultipleObjectiveLoss(nn.Module):
    r"""
    Weighted sum of losses.
    """

    def __init__(self, losses: List[_Loss] = tuple(), coefficients: List[float] = tuple()):
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

    def add_loss(self, loss: _Loss, weight: float) -> None:
        """
        :param loss: loss to add
        :param weight: weight of the loss
        """
        assert loss.reduction == "none", "The reduction of the loss must be none."

        self.losses.append(loss)
        self.coefficients.append(weight)

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


class GIoULoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Generalized Box Intersection over Union loss
        :param reduction: reduction method
        """
        super().__init__(reduction=reduction)
        self._giou_loss = generalized_box_iou_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :param target: ground truth boxes (num_pred, 4)
        :return: loss
        """
        return self._giou_loss(input, target, reduction=self.reduction)


class IoULoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Box Intersection over Union loss
        :param reduction: reduction method
        """
        super().__init__(reduction=reduction)
        self._iou_loss = distance_box_iou_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :param target: ground truth boxes (num_pred, 4)
        :return: loss
        """
        return self._iou_loss(input, target, reduction=self.reduction)


class SigmoidFocalLoss(_Loss):
    def __init__(self, reduction: str = "mean",
                 alpha: float = -1., gamma: float = 2.):
        """
        :reduction: reduction method
        :alpha: alpha parameter of the focal loss
        :gamma: gamma parameter of the focal loss
       """
        super(SigmoidFocalLoss, self).__init__(reduction=reduction)

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        :param preds: Predictions of the model (num_pred, num_classes)
        :param targets: Targets of the model (num_pred, num_classes)
        """
        prob = preds.sigmoid()
        p_t = prob * targets + (1 - prob) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        loss = loss.sum(dim=-1)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


# TODO: add the "focal loss" used in Def-DETR matching (?)


class SoftmaxNegLoss(_Loss):
    def __init__(self, reduction: str = "mean"):
        r"""
        Softmax negative loss.
        :param reduction: reduction method
        """
        super(SoftmaxNegLoss, self).__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        :param input: predicted scores (num_pred, num_classes)
        :param target: target label (num_pred, )
        """
        prob = input.softmax(dim=-1)
        loss = -prob[torch.arange(input.shape[0]), target]

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
