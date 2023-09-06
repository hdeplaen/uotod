from torch import Tensor

from ._POT import _POT

class UnbalancedPOT(_POT):
    available_methods = ['sinkhorn',
                         'sinkhorn_stabilized',
                         'sinkhorn_reg_scaling']

    def __init__(self,
                 method: str,
                 **kwargs) -> None:
        assert method in UnbalancedPOT.available_methods, \
            f"Only the following methods are available in the unbalanced case: {UnbalancedPOT.available_methods}"
        super(UnbalancedPOT, self).__init__(**{'balanced': False, **kwargs})

    def _matching(self, hist_pred: Tensor, hist_tgt: Tensor, C: Tensor, reg: float) -> Tensor:
        self._pot_method(a=hist_pred, b=hist_tgt, M=C, reg=reg, log=False, **self._method_kwargs)