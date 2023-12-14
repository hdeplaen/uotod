==============
Loss Functions
==============

Introduction
============

This package provides a comprehensive formulation of loss functions specifically designed for object detection tasks. It includes the general :class:`DetectionLoss <uotod.loss.DetectionLoss>` module, which serves as a foundation for the implementation of various loss functions. This module is designed to be flexible and extensible, allowing for the easy addition of new loss functions as needed.
This package also provides a set of loss functions that are commonly used in object detection tasks, such as the :class:`SigmoidFocalLoss <uotod.loss.SigmoidFocalLoss>` and the :class:`GIoULoss <uotod.loss.GIoULoss>`.

Different Losses
======

.. toctree::
    :maxdepth: 2

    detection
    multiple_objective
    modules

Examples
========

Commonly used loss functions
========

.. toctree::
    :maxdepth: 1

    detr
    def_detr
    ssd

