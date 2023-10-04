=======
Minimum
=======

.. include:: _preliminaries.rst

This class computed an exact minimum, either over the predictions, either over the targets.
Over the predictions (**source='prediction'**), the match :math:`\mathbf{P}` is given by

.. math::
    P_{i,j} = \Bigg\{
    \begin{array}{ll}
    1 & \text{if $j = \mathrm{arg\,min}_{j'}\left\{\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \varnothing\right)\right\}), \\
    0 & otherwise.
    \end{array}
    \Bigg\}

Similarly, the match over the targets (**source='target'**) is given by

.. math::
    P_{i,j} = \frac{\exp\left(\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_i,\mathbf{y}_j)\right)}{\sum_{k=1}^{N_p}\exp\left(\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_k,\mathbf{y}_j)\right)}.


Class
=====

.. autoclass:: uotod.match.Min
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

.. plot:: ../../example/match/min.py
    :include-source:


.. include:: _softmin_balanced.rst


.. include:: _min_balanced.rst


.. include:: _unmatched_to_background.rst