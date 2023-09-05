import torch
from torch import Tensor

from ._BalancedSinkhorn import _BalancedSinkhorn
from ._Compiled import _Compiled
from ..utils import extend_docstring


@extend_docstring(_BalancedSinkhorn)
@extend_docstring(_Compiled)
class BalancedSinkhorn(_BalancedSinkhorn, _Compiled):
    _compiled_name = "base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _matching(self, hist_pred: Tensor, hist_tgt: Tensor, C: Tensor, reg: float) -> Tensor:
        return self._matching_method(hist_pred, hist_tgt, C, reg, self._num_iter)

    def _matching_native(self, hist_pred: Tensor, hist_tgt: Tensor, C: Tensor, reg: float, num_iter: int) -> Tensor:
        batch_size, num_pred, _ = C.shape

        # Initialization of the algorithm
        K = torch.exp(-C / reg)
        u = torch.ones_like(hist_pred, requires_grad=False)

        # Iterations
        for _ in range(num_iter):
            u = hist_pred / (K * (hist_tgt / (K * u.unsqueeze(2)).sum(dim=1)).unsqueeze(1)).sum(dim=2)

        # Coupling matrix P = diag(u) @ K @ diag(v)
        P = torch.einsum("ni,nij,nj->nij", u, K, hist_tgt / (K * u.unsqueeze(2)).sum(dim=1))

        return P.data
