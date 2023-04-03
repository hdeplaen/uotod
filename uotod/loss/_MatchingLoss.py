from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss
from torch import Tensor


class _MatchingLoss(_Loss, ABCMeta):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        r"""
        :param size_average: is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        :param reduce: Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        :param reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        :type size_average: bool, optional
        :type reduce: bool, optional
        :type reduction: str, optional
        """
        super(_Loss).__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor):
        loss = self._loss(input, target)
        if self.reduction:
            # TODO: self.reduction is not a bool, and this may be different for various matching strategies (maybe leave it to the daughter classes?)
            num_pred = input.shape[0]
            return num_pred * loss
        return loss

    @abstractmethod
    def _loss(self, input: Tensor, target: Tensor):
        pass
