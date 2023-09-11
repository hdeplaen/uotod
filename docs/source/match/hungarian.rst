=========
Hungarian
=========

.. include:: _preliminaries.rst

The Hungarian matching :cite:`kuhn1955hungarian,munkres1957algorithmstransportationhungarian` minimizes the cost of the pairwise matches between the :math:`N_g` ground truth objects :math:`\left\{\mathbf{y}_j\right\}_{i=1}` with the :math:`N_p` predictions:

.. math::
    \hat{\sigma} = \mathrm{arg\, min} \left\{\sum_{j=1}^{N_g} \mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_{\sigma(j)},  \mathbf{y}_{j} \right): \sigma \in \mathcal{P}_{N_g}(\left[\!\left[ N_p \right]\!\right])\right\},

where :math:`\mathcal{P}_{N_g}(\left[\!\left[ N_p\right]\!\right]) = \left.\big\{\sigma \in \mathcal{P}(\left[\!\left[ N_p \right]\!\right]) \,\right| |\sigma| = N_g \big\}` is the set of possible combinations of :math:`N_g` in :math:`N_p`, with :math:`\mathcal{P}(\left[\!\left[ N_p \right]\!\right])` the power set of :math:`\left[\!\left[ N_p\right]\!\right]` (the set of all subsets). The match in matrix form :math:`\mathbf{P}`  itself is given by :math:`\hat{\sigma}(j) = \left\{ i : P_{i,j} \neq 0 \right\}`, or equivalently :math:`\hat{\sigma}(j) = \left\{ i : P_{i,j} = 1/N_p \right\}`

As an example, this matching strategy is used in the DETR family :cite:`carion2020detr,zhu2020deformabledetr`. The matching is one-to-one, minimizing the total cost. Every ground truth object is matched to a unique prediction, thus reducing the number of predictions needed. A downside is that the one-to-one matches may vary from one epoch to the next, again slowing down convergence :cite:`li2022dndetr,De_Plaen_2023_CVPR`.

.. note::
    Due its sequential nature, this matching strategy cannot make good use parallelization. In its SciPy implementation (default), it only runs on the CPU, moving back and forth the data if necessary.
    If regularization is not an issue (it often even helps with the convergence), we encourage to use :class:`uotod.match.BalancedSinkhorn` instead.


Class
=====

.. autoclass:: uotod.match.Hungarian
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

.. plot:: ../../example/hungarian.py
    :include-source:


Regularization with Balanced Sinkhorn
-----------------------------------
.. plot:: ../../example/balanced_vs_hungarian.py
    :include-source: