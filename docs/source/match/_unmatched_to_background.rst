Minimum over the targets
------------------------

For the argument **reg_pred=0**, the unbalanced case :class:`uotod.match.UnbalancedSinkhorn` behaves exactly the same
as the softmin :class:`uotod.match.SoftMin` with the same regularization and the targets as source. Indeed no marginal
distribution has to be satisfied over the predictions. As the background cost is uniform for all predictions, the
softmin over the background :math:`\varnothing` is totally uniform.

As the number of predictions is often fixed by design, but the number of actual objects to be predicted may vary for
each datapoint, the background :math:`\varnothing` is introduced. Its purpose is to become the output of any prediction
that is irrelevant for a specific datapoint, after training. Therefore, it does not make much sense to match a
prediction that is already matched to any non-background target, also to the background :math:`\varnothing`. In this
way, the uniform result on the background obtained by the softmin or the unbalanced case with **reg_pred=0** may not be
very useful: it would be better to only match the unmatched predictions (to any non-background target) to the background.

This result is obtained when considering the unbalanced case with **reg_pred** very low instead of zero, particularly
if the entropic regularization **reg** is also low. When the latter tends to zero, we recover an exact minimum
:class:`uotod.match.ClosestPrediction` from the targets. This justifies the argument **unmatched_to_background** of which the effect
can be visualized in the following example.


.. plot:: ../../example/match/uniform_background.py
    :include-source:


.. note::
    Considering the matching from the targets,
    if the uniform case over the background is seeked, we strongly encourage to use :class:`uotod.match.SoftMin` for
    a regularized result or :class:`uotod.match.ClosestPrediction` with **unmatched_to_background=False** for an unregularized
    example. This will always run faster than :class:`uotod.match.UnbalancedSinkhorn`.

    If the case where only the unmatched predictions are matched towards the background is seeked, we encourage to use
    :class:`uotod.match.UnbalancedSinkhorn` with a non-zero, but very low **reg_pred**, or :class:`uotod.match.ClosestPrediction` if
    no entropic regularization is seeked (with the default **unmatched_to_background=True**). This is unattainable with
    :class:`uotod.match.SoftMin`.

