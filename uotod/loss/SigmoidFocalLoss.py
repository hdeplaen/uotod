import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SigmoidFocalLoss(_Loss):
    r"""
    Creates a criterion that computes the Sigmoid Focal Loss.

    It was introduced in the paper `Focal Loss for Dense Object Detection`_.

    .. _Focal Loss for Dense Object Detection:
       https://arxiv.org/abs/1708.02002

    :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    :param alpha: Weighting factor alpha (default: -1.)
    :param gamma: Focusing parameter gamma (default: 2.)
    :type reduction: str, optional
    :type alpha: float, optional
    :type gamma: float, optional
    """
    def __init__(self, reduction: str = "mean", alpha: float = -1., gamma: float = 2.):
        super(SigmoidFocalLoss, self).__init__(reduction=reduction)

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, targets):
        r"""
        :param input: Predictions of the model (num_pred, num_classes)
        :param targets: Targets of the model (num_pred, num_classes)
        :type input: Tensor (float)
        :type targets: Tensor (float)
        :return: loss
        :rtype: Tensor (float)
        """
        prob = input.sigmoid()
        p_t = prob * targets + (1 - prob) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(input, targets, reduction="none")
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
