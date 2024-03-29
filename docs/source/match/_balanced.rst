The Optimal Transport problem aims at minimizing the transport cost between a source distribution (the predictions) and a target distribution :cite:`monge1781memoire,kantorovich,villani2009optimal`.
This particular problem solves a regularization of the original problem with Sinkhorn's algorithm, well suited for GPU parallelization :cite:`cuturi2013sinkhorn,peyre2019computational`.
In the limit of **reg** (or **reg_dimless**) to 0, this becomes the Hungarian algorithm. We refer to Proposition 1 of :cite:`De_Plaen_2023_CVPR` for further information. By consequence, this matching can be seen as a regularized version of the Hungarian one.

.. include:: _preliminaries.rst

For each element in the batch, the following problem is solved and the match :math:`\mathbf{P}` is retrieved.

.. math::
    :nowrap:

    \begin{align}
    N_p * \underset{\mathbf{P}\in \mathbb{R}^{N_p \times N_t+1}}{\mathrm{arg\,min}} &\sum_{i,j=1}^{N_p,N_t+1} P_{i,j}\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \mathbf{y}_j\right) - \mathtt{reg} * \,\mathrm{H}(\mathbf{P}), &\\
    \mathrm{s.t.} & \sum_{j=1}^{N_t+1} P_{i,j} = 1/N_p, & \forall\; 0 \leq i \leq N_p \;\text{(predictions)},\\
    & \sum_{i=1}^{N_p} P_{i,j} = 1/N_p, & \forall\; 0 \leq j \leq N_t \;\text{(targets)},\\
    & \sum_{i=1}^{N_p} P_{i,j} = (N_p - N_t)/N_p, & j = N_t+1\;\text{(background)}.
    \end{align}

with :math:`\mathrm{H}: \Delta^{N \times M} \rightarrow \mathbb{R}_{\geq 0} : \mathbf{P} \mapsto -\sum_{i,j} P_{i,j}(\log(P_{i,j})-1)` the entropy of the match :math:`\mathbf{P}`, with :math:`0 \ln(0) = 0` by definition.

In the particular case where no background is used, the problem remains the same but the last column of :math:`\mathbf{P}` is just unnused.

.. warning::
    If the formulation converges to the Hungarian algorithm in the limit of **reg** (or **reg_dimless**) to 0, it becomes more and more unstable if solved using Sinkhorn's algorithm. We encourage to use :class:`uotod.match.Hungarian` if no regularization at all is explicitly seeked, at the cost of loosing parallelization.
