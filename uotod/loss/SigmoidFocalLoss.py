from torch.nn.modules.loss import _Loss

class SigmoidFocalLoss(_Loss):
    def __init__(self, reduction: str = "mean",
                 alpha: float = -1., gamma: float = 2.):
        """
        :reduction: reduction method
        :alpha: alpha parameter of the focal loss
        :gamma: gamma parameter of the focal loss
       """
        super(SigmoidFocalLoss, self).__init__(reduction=reduction)

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        :param preds: Predictions of the model (num_pred, num_classes)
        :param targets: Targets of the model (num_pred, num_classes)
        """
        prob = preds.sigmoid()
        p_t = prob * targets + (1 - prob) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
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
