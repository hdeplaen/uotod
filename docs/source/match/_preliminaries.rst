We consider the matching cost :math:`\mathcal{L}_{\text{match}}` = **cls_match_cost** + **loc_match_cost**
between the :math:`N_p` predictions :math:`\hat{\mathbf{y}}_i` and :math:`N_t` targets :math:`\mathbf{y}_j`. In particular, the cost of the background :math:`\mathbf{y}_{N_t+1} = \varnothing` is given by
:math:`\mathcal{L}_{\text{match}}\left(\hat{\mathbf{y}}_i, \varnothing\right)` = **bg_cost**.
The match is given by