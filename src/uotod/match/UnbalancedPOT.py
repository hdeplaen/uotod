from typing import Optional

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

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, **kwargs):
        return self._matching_method(a=kwargs['hist_pred'],
                                               b=kwargs['hist_target'],
                                               M=cost_matrix,
                                               reg=kwargs['reg'],
                                               reg_m=self.reg_pred_target,
                                               **self._method_kwargs)
