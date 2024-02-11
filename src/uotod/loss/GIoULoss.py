import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


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

    def _giou(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :type input: Tensor (float)
        :param target: ground truth boxes (num_pred, 4)
        :type target: Tensor (float)
        :return: GIoU
        :rtype: Tensor (float)
        """

        x1 = torch.max(input[:, 0], target[:, 0])
        y1 = torch.max(input[:, 1], target[:, 1])
        x2 = torch.min(input[:, 2], target[:, 2])
        y2 = torch.min(input[:, 3], target[:, 3])

        intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)
        union = (input[:, 2] - input[:, 0]) * (input[:, 3] - input[:, 1]) + \
                (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]) - intersection

        iou = intersection / union

        x1 = torch.min(input[:, 0], target[:, 0])
        y1 = torch.min(input[:, 1], target[:, 1])
        x2 = torch.max(input[:, 2], target[:, 2])
        y2 = torch.max(input[:, 3], target[:, 3])

        enclose_area = (x2 - x1) * (y2 - y1)

        giou = iou - (enclose_area - union) / enclose_area

        return giou

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
