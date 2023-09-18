==============
Detection Loss
==============

This module contains the loss function for object detection.
The loss function is computed in two steps:

* First, a match is determined between predicted and ground truth boxes. The match is computed by the matching_method module.
* Then, the loss is calculated as a weighed sum of the classification and regression losses.

The loss is implemented as a PyTorch module, so it can be used as a loss function in a PyTorch training loop.

Class
=====

.. autoclass:: uotod.loss.DetectionLoss
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:
