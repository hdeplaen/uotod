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

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, **kwargs):
        return self._matching_method(a=kwargs['hist_pred'],
                                     b=kwargs['hist_target'],
                                     M=cost_matrix,
                                     reg=kwargs['reg'],
                                     **self._method_kwargs)
