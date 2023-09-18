from torch import Tensor
from torch.nn.modules.loss import _Loss
from torchvision.ops import distance_box_iou_loss


class IoULoss(_Loss):
    r"""
    Creates a criterion that measures the IoU loss between each predicted box and target box.

    It is a wrapper around the `box_iou_loss` function from the `torchvision` package.

    :param reduction: Specifies the reduction to apply to the output:
    :type reduction: str, optional
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self._iou_loss = distance_box_iou_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :type input: Tensor (float)
        :param target: ground truth boxes (num_pred, 4)
        :type target: Tensor (float)
        :return: loss
        :rtype: Tensor (float)
        """

        return self._iou_loss(input, target, reduction=self.reduction)