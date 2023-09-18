import torch
from torch import Tensor

from ._POT import _POT
from ._Sinkhorn import _Sinkhorn
from ..utils import kwargs_decorator, extend_docstring


class BalancedPOT(_POT, _Sinkhorn):
    available_methods = ['sinkhorn',
                         'greenkorn',
                         'sinkhorn_log',
                         'sinkhorn_stabilized',
                         'sinkhorn_epsilon_scaling']

    @kwargs_decorator({'method': 'sinkhorn'})
    def __init__(self, **kwargs) -> None:
        assert kwargs["method"] in BalancedPOT.available_methods, \
            f"Only the following methods are available in the balanced case: {BalancedPOT.available_methods}"
        super(BalancedPOT, self).__init__(**{'balanced': True, **kwargs})

    def _matching(self, hist_pred: Tensor, hist_tgt: Tensor, C: Tensor, reg: float) -> Tensor:
        sols = []
        for idx in range(C.size(0)):
            hp = hist_pred[idx, :]
            ht = hist_tgt[idx, :]
            C_loc = C[idx, :, :]
            sols.append(
                self._pot_method(a=hp,
                                 b=ht,
                                 M=C_loc,
                                 reg=reg,
                                 **self._method_kwargs).unsqueeze(0)
            )
        return torch.cat(sols, dim=0)



