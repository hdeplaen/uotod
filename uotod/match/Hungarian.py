from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from scipy.optimize import linear_sum_assignment

from ._Match import _Match
from ..utils import extend_docstring

@extend_docstring(_Match)
class Hungarian(_Match):
    r"""
    The Hungarian matching :cite:`kuhn1955hungarian,munkres1957algorithmstransportationhungarian` minimizes the cost of the pairwise matches between the :math:`N_g` ground truth objects :math:`\left\{\mathbf{y}_j\right\}_{i=1}` with the :math:`N_p` predictions:

    .. math::
        \hat{\sigma} = \mathrm{arg\, min} \left\{\sum_{j=1}^{N_g} \mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_{\sigma(j)},  \mathbf{y}_{j} \right): \sigma \in \mathcal{P}_{N_g}(\left[\!\left[ N_p \right]\!\right])\right\},

    where :math:`\mathcal{P}_{N_g}(\left[\!\left[ N_p\right]\!\right]) = \left.\big\{\sigma \in \mathcal{P}(\left[\!\left[ N_p \right]\!\right]) \,\right| |\sigma| = N_g \big\}` is the set of possible combinations of :math:`N_g` in :math:`N_p`, with :math:`\mathcal{P}(\left[\!\left[ N_p \right]\!\right])` the power set of :math:`\left[\!\left[ N_p\right]\!\right]` (the set of all subsets).

    As an example, this matching strategy is used in the DETR family :cite:`carion2020detr,zhu2020deformabledetr`. The matching is one-to-one, minimizing the total cost. Every ground truth object is matched to a unique prediction, thus reducing the number of predictions needed. A downside is that the one-to-one matches may vary from one epoch to the next, again slowing down convergence :cite:`li2022dndetr`.

    .. note::
        Due its sequential nature, this matching strategy cannot make good use parallelization, i.e. GPU architectures. It therefore only runs on the CPU, moving back and forth the data if necessary.

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @extend_docstring(_Match.compute_matching)
    @torch.no_grad()
    def compute_matching(self, cost_matrix: Tensor, target_mask: Tensor) -> Tensor:
        r"""
        This method makes use of Scipy's `linear_sum_assignment`.
        """
        device = cost_matrix.device
        # Move to CPU
        cost_matrix = cost_matrix.cpu()

        # Compute the matching
        matching = torch.zeros_like(cost_matrix)
        for i in range(cost_matrix.shape[0]):
            num_tgt = target_mask[i].sum().item()
            row_ind, col_ind = linear_sum_assignment(cost_matrix[i, :, :num_tgt].numpy())
            matching[i, row_ind, col_ind] = 1.

        # Move back to GPU
        matching = matching.to(device)

        # Assign unmatched predictions to the background
        matching[:, :, -1] = 1. - matching[:, :, :-1].sum(dim=2)

        return matching