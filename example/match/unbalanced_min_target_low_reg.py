import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_min_pred = uotod.match.Min(loc_match_module=L, background_cost=0.8, source='target')
M_unb_small = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01, reg_pred=1.e-2, reg_target=1.e+4)
M_unb_med = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01, reg_pred=.2, reg_target=1.e+4)
M_unb_big = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01, reg_pred=1.e+4, reg_target=1.e+4)
M_balanced = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01)
M_hungarian = uotod.match.Hungarian(loc_match_module=L, background_cost=0.8)


matches = [M_min_pred(input, target)[0, :, :],
           M_unb_small(input, target)[0, :, :],
           M_unb_med(input, target)[0, :, :],
           M_unb_big(input, target)[0, :, :],
           M_balanced(input, target)[0, :, :],
           M_hungarian(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['Minimum from\nthe predictions',
                                                     'Unbalanced Sink.\n(low reg_pred)',
                                                     'Unbalanced Sink.\n(medium reg_pred)',
                                                     'Unbalanced Sink.\n(high reg_pred)',
                                                     'Balanced\nSinkhorn',
                                                     'Hungarian\nAlgorithm'],
                                          title='Effect of reg_pred (reg=0.01)',
                                          figsize=(24, 7))
fig_matches.show()
