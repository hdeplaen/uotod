import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SigmoidFocalLoss(_Loss):
    r"""
    Creates a criterion that computes the Sigmoid Focal Loss.

    .. math::
        loss(x, y) = \sum_{i=1}^{N_c} \left( y_i \alpha (1 - p_i)^{\gamma} (-log(p_i)) + (1 - y_i) (1 - \alpha) p_i^{\gamma} (-log(1 - p_i)) \right)

    where :math:`p_i = \sigma(x_i)` is the probability of class i, :math:`y \in \{0,1\}^{N_c}` is the target label in one-hot encoding,
    :math:`N_c` is the number of classes, :math:`\gamma` is the focusing parameter
    and :math:`\alpha` is the weighting factor.


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
        :param input: Predictions (num_pred, num_classes)
        :param targets: Targets in one-hot encoding (num_pred, num_classes)
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

        loss = loss.sum(dim=-1)  # sum over the classes

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class SigmoidFocalCost(_Loss):
    r"""
    Creates a criterion that computes the Sigmoid Focal Cost.

    It is a variant of the Sigmoid Focal Loss, where both a negative and a positive cost term are used.
    Unlike the original focal loss, the loss is computed only for the target class.

    .. math::
        cost(x, y) = \alpha (1 - p_y)^{\gamma} (-log(p_y)) - (1 - \alpha) p_y^{\gamma} (-log(1 - p_y))

    where :math:`p_y = \sigma(x_y)` is the probability of the target class y, :math:`\gamma` is the focusing parameter, and
    :math:`\alpha` is the weighting factor.

    It was introduced in the paper :cite:`zhu2020deformabledetr`.

    :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    :param alpha: Weighting factor alpha (default: -1.)
    :param gamma: Focusing parameter gamma (default: 2.)
    :type reduction: str, optional
    :type alpha: float, optional
    :type gamma: float, optional
    """
    def __init__(self, reduction: str = "mean", alpha: float = -1., gamma: float = 2.):
        super(SigmoidFocalCost, self).__init__(reduction=reduction)

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, targets):
        r"""
        :param input: Predictions (num_pred, num_classes)
        :param targets: Targets labels (num_pred,) or one-hot encoding (num_pred, num_classes)
        :type input: Tensor (float)
        :type targets: Tensor (int) or Tensor (float)
        :return: loss
        :rtype: Tensor (float)
        """
        assert input.dim() == 2, "The input must be of shape (num_pred, num_classes)."
        assert targets.dim() in [1, 2], "The targets must be of shape (num_pred,) or (num_pred, num_classes)."

        # convert targets to labels if they are in one-hot encoding, if needed
        if targets.dim() == input.dim():
            targets = targets.argmax(dim=-1).to(torch.long)

        prob = input.sigmoid()
        prob = prob[torch.arange(len(targets)), targets]  # keep only the probability of the target class

        pos_cost = ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
        neg_cost = (prob ** self.gamma) * (-(1 - prob + 1e-8).log())

        loss = pos_cost - neg_cost

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
