=======
SoftMin
=======

Definition
==========

.. include:: _preliminaries.rst

This class computed a soft minimum, either over the predictions, either over the targets.
Over the predictions (**source='prediction'**), the match :math:`\mathbf{P}` is given by

.. math::
    P_{i,j} = \frac{\exp\left(-\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_i,\mathbf{y}_j)\right)}{\sum_{k=1}^{N_t+1}\exp\left(-\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_i,\mathbf{y}_k)\right)}.

Similarly, the match over the targets (**source='target'**) is given by

.. math::
    P_{i,j} = \frac{\exp\left(-\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_i,\mathbf{y}_j)\right)}{\sum_{k=1}^{N_p}\exp\left(-\mathcal{L}_{\mathrm{match}}(\hat{\mathbf{y}}_k,\mathbf{y}_j)\right)}.

This is essentially the opposite of a softmax: :math:`\mathrm{softmin}(\bf{x}) = \mathrm{softmax}(-\bf{x})`.

Module
======

.. autoclass:: uotod.match.SoftMin
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

Simple Example
--------------

.. plot:: ../../example/match/softmin.py
    :include-source:

.. include:: _min_balanced.rst

.. include:: _unmatched_to_background.rst

.. include:: _min_softmin.rst