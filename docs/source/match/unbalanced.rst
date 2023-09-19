===================
Unbalanced Sinkhorn
===================

.. include:: _preliminaries.rst

.. math::
    N_p * \underset{\mathbf{P} \in \mathbb{R}_{\geq 0}^{N_p\times N_t+1}}{\mathrm{arg\, min}} \bigg\{ \mathtt{reg}*\mathrm{KL}\left(\mathbf{P}|\mathbf{K}_{\mathtt{reg}}\right)\, + &\,\mathtt{reg\_pred}*\mathrm{KL}\left(\mathbf{P}\mathbf{1}_{N_t+1}|\mathbf{\alpha}\right) \\
    + &\,\mathtt{reg\_target}*\mathrm{KL}\left(\mathbf{1}_{N_p}^\top \mathbf{P}|\mathbf{\beta}\right)\bigg\},

where :math:`\mathrm{KL}: \mathbb{R}^{N\times M}_{\geq 0} \times \mathbb{R}^{N\times M}_{> 0} \rightarrow \mathbb{R}_{\geq 0}^{\phantom{N}} : (\mathbf{U},\mathbf{V}) \mapsto \sum_{i,j=1}^{N \times M} U_{i,j} \log(U_{i,j} / V_{i,j}) - U_{i,j} + V_{i,j}` is the `Kullback-Leibler divergence` -- also called `relative entropy` -- between matrices or vectors when :math:`M=1`, with :math:`0 \ln (0) = 0` by definition. The Gibbs kernel :math:`\mathbf{K}_{\mathtt{reg}}` is given by :math:`\left(K_{\mathtt{reg}}\right)_{i,j} = \exp\left(- \mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_j \right)/ \mathtt{reg} \right)`.

The marginals for the soft constraints are given by

.. math::
    :nowrap:

    \begin{eqnarray}
        \mathbf{\alpha} &=& \frac{1}{N_p}[\; \underbrace{1, \; 1, \;1,\; \ldots, \; 1}_{\text{$N_p$ predictions}}\;], \\
        \mathbf{\beta} &=& \frac{1}{N_p}[\; \underbrace{1, \; 1, \; \ldots, \; 1}_{\text{$N_t$ targets}}, \; \underbrace{(N_p-N_t)}_{\text{background }\varnothing} \;].
    \end{eqnarray}

In the particular case where no background is used, the problem remains the same but the last column of :math:`\mathbf{P}` is just unnused.


.. warning::
    If the formulation converges to the Hungarian algorithm in the limit of **reg** (or **reg_dimless**) to 0, it becomes more and more unstable if solved using Sinkhorn's algorithm.

A variation of Sinkhorn's algorithm is used to solve the problem.
It is however not run until convergence, but with a fixed number of iterations.
A compiled version is also available through the boolean `compiled`.
The fixed number of iterations (and the non-verification of convergence) has the advantage of boosting the computation time and taking a vast advantage of the Tensor parallelization capabilities of modern GPUs.
If a more stable or defensive, but also slower implementation is required, we refer to the :class:`uotod.match.UnbalancedPOT` class. It however uses the same parameter for both soft constraints.
In other words, it only works for the cases where **reg_pred** = **reg_target**.



Class
=====

.. autoclass:: uotod.match.UnbalancedSinkhorn
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:


Example
=======

The following shows the limit cases of Unbalanced Sinkhorn. As each limit case comes with its own algorithm, it is preferrable to use that one for efficiency. They are provided here as an illustration to better understand the role of each parameter. The goal of this algorithm is compute hybrid cases between the limit cases that are unattainable otherwise.

.. plot:: ../../example/match/unbalanced.py
    :include-source:

Limit Cases
===========

