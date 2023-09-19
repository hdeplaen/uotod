from torch import Tensor
from torchvision.ops import generalized_box_iou_loss
from torch.nn.modules.loss import _Loss


class GIoULoss(_Loss):
    r"""
    Creates a criterion that measures the generalized IoU loss between each predicted box and target box.

    It is a wrapper around the `generalized_box_iou_loss` function from the `torchvision` package.
    """
    def __init__(self) -> None:
        super().__init__()
        self._giou_loss = generalized_box_iou_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :type input: Tensor (float)
        :param target: ground truth boxes (num_pred, 4)
        :type target: Tensor (float)
        :return: loss
        :rtype: Tensor (float)
        """
        return self._giou_loss(input, target)