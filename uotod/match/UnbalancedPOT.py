import torch
from torch import Tensor

from ._POT import _POT
from ._Sinkhorn import _Sinkhorn
from ..utils import kwargs_decorator, extend_docstring

@extend_docstring(_Sinkhorn)
class UnbalancedPOT(_POT, _Sinkhorn):
    r"""
    :param method:
    :type method:
    :param reg_pred_target: Defaults to 1.
    :type reg_pred_target: float, optional
    """

    available_methods = ['sinkhorn',
                         'sinkhorn_stabilized',
                         'sinkhorn_reg_scaling']

    @kwargs_decorator({'method': 'sinkhorn_knopp_unbalanced',
                       'reg_pred_target': 1.})
    def __init__(self, **kwargs) -> None:
        assert kwargs['method'] in UnbalancedPOT.available_methods, \
            f"Only the following methods are available in the unbalanced case: {UnbalancedPOT.available_methods}"
        super(UnbalancedPOT, self).__init__(**{'balanced': False, **kwargs})
        self.reg_pred_target = kwargs['reg_pred_target']

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
                                 reg_m=self.reg_pred_target,
                                 **self._method_kwargs) \
                    .unsqueeze(0)
            )
        return torch.cat(sols, dim=0)