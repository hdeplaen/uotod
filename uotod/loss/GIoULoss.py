from torch import Tensor
from torch.nn.modules.loss import _Loss
from torchvision.ops.boxes import generalized_box_iou


class GIoULoss(_Loss):
    r"""
    Creates a criterion that measures the generalized IoU loss between each predicted box and target box.

    .. math::
        \text{loss} = 1 - \text{GIoU}

    :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    :type reduction: str, optional
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self._giou = generalized_box_iou

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :type input: Tensor (float)
        :param target: ground truth boxes (num_pred, 4)
        :type target: Tensor (float)
        :return: loss
        :rtype: Tensor (float)
        """

        loss = 1.0 - self._giou(input, target)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
