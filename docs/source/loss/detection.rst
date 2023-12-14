==============
Detection Loss
==============

This module contains the loss function for object detection.
The loss function is computed in two steps:

1. A match :math:`\mathbf{P}` is determined between predicted and ground truth boxes. The match is computed by the **matching_method** module. See the `Matching Strategies <../match>`_ section for more details.
2. The loss is calculated as a weighed sum of the classification and bounding box regression losses: :math:`\mathcal{L}_{\text{train}}(\hat{\mathbf{y}}_i,\mathbf{y}_j) = \mathcal{L}_{\text{classification}}(\hat{\mathbf{c}}_i,\mathbf{c}_j) + \mathcal{L}_{\text{localization}}(\hat{\mathbf{b}}_i,\mathbf{b}_j)` between the :math:`N_p` predictions :math:`\hat{\mathbf{y}}_i` and :math:`N_g` targets :math:`\mathbf{y}_j`. The particular training loss for the background ground truth includes only a classification term :math:`\mathcal{L}_{\text{train}}(\hat{\mathbf{y}}_i,\varnothing) = \mathcal{L}_{\text{classification}}(\hat{\mathbf{c}}_i,\varnothing)`.

.. math::
    :nowrap:

    \begin{align}
    loss = N_p \sum_{i=1}^{N_p} \sum_{j=1}^{N_g+1} \hat{P}_{ij} \mathcal{L}_{\text{train}}(\hat{\mathbf{y}}_i,\mathbf{y}_j)
    \end{align}


- If reduction is set to ``'mean'``, the localization loss is divided by the number of matched predictions :math:`N_{pos} = N_p \sum_{i=1}^{N_p} \sum_{j=1}^{N_g} \hat{P}_{ij}` and the classification loss is divided by the total number of predictions :math:`N_{\text{tot}} = N_p\sum_{i=1}^{N_p} \sum_{j=1}^{N_g+1} \hat{P}_{ij} \approx N_p`.
- If reduction is set to ``'sum'``, the loss is not divided by any factor.

The loss is implemented as a PyTorch module, so it can be used as a loss function in a PyTorch training loop.

Class
=====

.. autoclass:: uotod.loss.DetectionLoss
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:
