From the Closest Target to the Hungarian Algorithm
--------------------------------------------------

The module :class:`uotod.match.UnbalancedSinkhorn` with low regularization can play the role of an interpolant between
:class:`uotod.match.ClosestTarget` and :class:`uotod.match.Hungarian` (or :class:`uotod.match.BalancedSinkhorn` with the same low
regularization).

A high **reg_pred** will enforce a strong respect of the mass constraints on the predictions. If **reg_target** is
close to zero, this will emulate a minimum as the problem essentially minimizes the objective for each prediction,
disregarding the mass constraints on the targets. For a high **reg_target**, the problem will essentially minimize the
same objective as the :class:`uotod.match.BalancedSinkhorn`, which approximates the :class:`uotod.match.Hungarian` with
a low regularization. This is illustrated in the following example.

.. plot:: ../../example/match/unbalanced_min_target_low_reg.py
    :include-source:

When a edge case is seeked after--either :class:`uotod.match.ClosestTarget` or :class:`uotod.match.Hungarian`--, we encourage
to directly use these modules instead of the module :class:`uotod.match.UnbalancedSinkhorn`, which is slower in
computation time. The latter should only be used when seeking for an in-between case.

.. note::
    Similarly, when a higher regularization is used, the module :class:`uotod.match.UnbalancedSinkhorn` plays the role of an
    interpolant between a :class:`uotod.match.SoftMin` and a :class:`uotod.match.BalancedSinkhorn` with the same
    regularization.

.. note::
    The opposite case with a high **reg_target** will approximate :class:`uotod.match.ClosestPrediction`.