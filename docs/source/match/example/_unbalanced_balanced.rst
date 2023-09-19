Approximation of the Balanced Case and the Hungarian Algorithm
--------------------------------------------------------------

This example shows how increasing the both constraint regularization parameter tends to the Balanced case. If this limit case is seeked after, it is preferrable to use :class:`uotod.match.BalancedSinkhorn` instead.

.. plot:: ../../example/match/unbalanced_vs_balanced.py
    :include-source:


If lowering the regularization, we also recover the Hungarian algorithm.