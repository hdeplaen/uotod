=============================================================
Unbalanced Sinkhorn from the Python Optimal Transport Library
=============================================================

.. include:: _preliminaries.rst

.. math::
    N_p * \underset{\mathbf{P} \in \mathbb{R}_{\geq 0}^{N_p\times N_t+1}}{\mathrm{arg\, min}} \bigg\{ \mathtt{reg}*\mathrm{KL}\left(\mathbf{P}|\mathbf{K}_{\mathtt{reg}}\right)\, + &\,\mathtt{reg\_pred\_target}*\mathrm{KL}\left(\mathbf{P}\mathbf{1}_{N_t+1}|\mathbf{\alpha}\right) \\
    + &\,\mathtt{reg\_pred\_target}*\mathrm{KL}\left(\mathbf{1}_{N_p}^\top \mathbf{P}|\mathbf{\beta}\right)\bigg\},

where :math:`\mathrm{KL}: \mathbb{R}^{N\times M}_{\geq 0} \times \mathbb{R}^{N\times M}_{> 0} \rightarrow \mathbb{R}_{\geq 0}^{\phantom{N}} : (\mathbf{U},\mathbf{V}) \mapsto \sum_{i,j=1}^{N \times M} U_{i,j} \log(U_{i,j} / V_{i,j}) - U_{i,j} + V_{i,j}` is the `Kullback-Leibler divergence` -- also called `relative entropy` -- between matrices or vectors when :math:`M=1`, with :math:`0 \ln (0) = 0` by definition. The Gibbs kernel :math:`\mathbf{K}_{\mathtt{reg}}` is given by :math:`\left(K_{\mathtt{reg}}\right)_{i,j} = \exp\left(- \mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_j \right)/ \mathtt{reg} \right)`.

The marginals for the soft constraints are given by

.. math::
    :nowrap:

    \begin{eqnarray}
        \mathbf{\alpha} &=& \frac{1}{N_p}[\; \underbrace{1, \; 1, \;1,\; \ldots, \; 1}_{\text{$N_p$ predictions}}\;], \\
        \mathbf{\beta} &=& \frac{1}{N_p}[\; \underbrace{1, \; 1, \; \ldots, \; 1}_{\text{$N_t$ targets}}, \; \underbrace{(N_p-N_t)}_{\text{background }\varnothing} \;].
    \end{eqnarray}

In the particular case where no background is used, the problem remains the same but the last column of :math:`\mathbf{P}` is just unnused.

.. note::
    Unlike :class:`uotod.match.UnbalancedSinkhorn`, the same regularization parameter is used for both the predictions and the targets.

.. warning::
    If the formulation converges to the Hungarian algorithm in the limit of `reg` (or `reg_dimless`) to 0, it becomes more and more unstable if solved using Sinkhorn's algorithm.


Class
=====

.. autoclass:: uotod.match.UnbalancedPOT
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members:



Available Methods
=================

+-----------+------------------------+--------------------------+------------+
| Value     | Description            | Extra Optional Arguments | Reference  |
+===========+========================+==========================+============+
|           |                        |                          |            |
+-----------+------------------------+--------------------------+------------+