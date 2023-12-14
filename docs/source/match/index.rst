===================
Matching Strategies
===================

Introduction
============
This package provides implementations of various matching strategies, all of which are particular cases of Optimal Transport. These strategies are crucial in object detection tasks, as they determine how predictions are matched with ground truth objects. The module provides a range of strategies, including :doc:`ClosestTarget <closest_target>` (used in methods like Faster R-CNN and SSD), :doc:`Hungarian <hungarian>` (used in DETR and its extensions) and :doc:`UnbalancedSinkhorn <unbalanced>` (which generalizes the other strategies). Each strategy is implemented as a separate class, allowing for easy extension and customization.

Different Strategies
====================

.. toctree::
    :maxdepth: 1

    hungarian
    balanced
    unbalanced
    closest_prediction
    closest_target
    min
    softmin
    balanced_pot
    unbalanced_pot