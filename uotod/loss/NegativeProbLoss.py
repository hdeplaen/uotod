import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class NegativeProbLoss(_Loss):
    r"""
    Creates a criterion that computes the negative probability of the target class.

    .. math::
        \text{loss}(x, y) = -softmax(x)[y]

    :param reduction: reduction method
    :type reduction: str, optional
    """
    def __init__(self, reduction: str = "mean"):
        super(NegativeProbLoss, self).__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        :param input: Predicted scores (num_pred, num_classes)
        :param target: Ground-truth label (num_pred, )
        :type input: Tensor (float)
        :type target: Tensor (long)
        :return: loss
        :rtype: Tensor (float)
        """
        prob = input.softmax(dim=-1)
        loss = -prob[torch.arange(input.shape[0]), target]

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
