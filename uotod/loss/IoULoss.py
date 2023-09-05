from torch import Tensor
from torch.nn.modules.loss import _Loss
from torchvision.ops import distance_box_iou_loss

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