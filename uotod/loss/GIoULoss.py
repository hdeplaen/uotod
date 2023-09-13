from torch import Tensor
from torchvision.ops import generalized_box_iou_loss
from torch.nn.modules.loss import _Loss


class GIoULoss(_Loss):
    def __init__(self) -> None:
        """
        Generalized Box Intersection over Union loss
        :param reduction: reduction method
        """
        super().__init__(reduction='none')
        self._giou_loss = generalized_box_iou_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: predicted boxes (num_pred, 4)
        :param target: ground truth boxes (num_pred, 4)
        :return: loss
        """
        return self._giou_loss(input, target, reduction=self.reduction)