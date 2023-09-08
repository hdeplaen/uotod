=================
Balanced Sinkhorn
=================

.. math::
    \hat{\mathbf{P}} = \underset{\mathbf{P}\, \in\, \mathcal{U}(\mathbf{\alpha},\mathbf{\beta})}{\mathrm{arg\,min}} \left\{\sum_{i,j=1}^{N_p,N_g} P_{i,j}\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_j\right) - \mathtt{reg} * \,\mathrm{H}(\mathbf{P}) \right\},

.. warning::
    If the formulation converges to the Hungarian algorithm in the limit of `reg` to 0, it becomes more and more unstable if solved using Sinkhorn's algorithm.

Class
=====

.. autoclass:: uotod.match.BalancedSinkhorn
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

.. plot:: ../../example/balanced.py
    :include-source:

This is in essence a regularized version of the Hungarian algorithm.

.. plot:: ../../example/balanced_vs_hungarian.py
    :include-source: