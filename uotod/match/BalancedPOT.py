from torch import Tensor

from ._POT import _POT
from ._BalancedSinkhorn import _BalancedSinkhorn
from ..utils import kwargs_decorator, extend_docstring


class BalancedPOT(_POT, _BalancedSinkhorn):
    available_methods = ['sinkhorn',
                         'greenkorn',
                         'screenkorn',
                         'sinkhorn_log',
                         'sinkhorn_stabilized']

    @kwargs_decorator({'method': 'sinkhorn'})
    def __init__(self, **kwargs) -> None:
        assert kwargs["method"] in BalancedPOT.available_methods, \
            f"Only the following methods are available in the balanced case: {BalancedPOT.available_methods}"
        super(BalancedPOT, self).__init__(**{'balanced': True, **kwargs})

    def _matching(self, hist_pred: Tensor, hist_tgt: Tensor, C: Tensor, reg: float) -> Tensor:
        self._pot_method(a=hist_pred, b=hist_tgt, M=C, reg=reg, log=False, **self._method_kwargs)
