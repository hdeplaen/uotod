==============
Closest Target
==============

.. include:: _preliminaries.rst

This class computes an exact minimum over the predictions, in other words it matches each prediction to the closest target.
The match :math:`\mathbf{P}` is given by

.. math::
    P_{i,j} = \Bigg\{
    \begin{array}{ll}
    1 & \text{if $j = \mathrm{arg\,min}_{k \in [1,N_t+1]}\left\{\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_{k}\right)\right\}$}, \\
    0 & \text{otherwise}.
    \end{array}

For the opposite where each target is matched towards the closest prediction, we refer to :class:`uotod.match.ClosestPrediction`.

Class
=====

.. autoclass:: uotod.match.ClosestTarget
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

.. plot:: ../../example/match/closest_target.py
    :include-source:


.. include:: _softmin_balanced.rst


.. include:: _min_balanced_target.rst


.. include:: _unmatched_to_background.rst