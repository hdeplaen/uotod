==================
Closest Prediction
==================

.. include:: _preliminaries.rst

This class computes an exact minimum over the targets, in other words it matches each prediction to the closest target.
The match :math:`\mathbf{P}` is given by for the first :math:`N_t` targets.

.. math::
    P_{i,j} = \Bigg\{
    \begin{array}{ll}
    1 & \text{if $i = \mathrm{arg\,min}_{k \in [1,N_p]}\left\{\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_k, \mathbf{y}_j\right)\right\}$) and $\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_j\right) \leq \text{threshold}$}, \\
    0 & \text{otherwise}.
    \end{array}

For the background :math:`N_t+1`, it is either uniform, either :math:`1` for all unmatched predictions and :math:`0`
for the others, depending on the parameter **uniform_background** (see further).

For the opposite where each target is matched towards the closest prediction, we refer to :class:`uotod.match.ClosestTarget`.

Class
=====

.. autoclass:: uotod.match.ClosestPrediction
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

.. plot:: ../../example/match/closest_prediction.py
    :include-source:


.. include:: _softmin_balanced.rst


.. include:: _min_balanced_pred.rst