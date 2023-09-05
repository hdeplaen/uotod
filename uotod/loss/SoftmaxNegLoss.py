import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

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