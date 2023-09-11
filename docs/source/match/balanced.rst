=================
Balanced Sinkhorn
=================

.. include:: _balanced.rst

Sinkhorn's algorithm is used to solve the problem.
It is however not run until convergence, but with a fixed number of iterations.
A compiled version is also available through the boolean `compiled`.
The fixed number of iterations (and the non-verification of convergence) has the advantage of boosting the computation time and taking a vast advantage of the Tensor parallelization capabilities of modern GPUs.
If a more stable or defensive, but also slower implementation is required, we refer to the :class:`uotod.match.BalancedPOT` class.

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