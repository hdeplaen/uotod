import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_softmin_pred = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.1, source='prediction')
M_unb_small = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.1, reg_pred=1.e+4, reg_target=1.e-2)
M_unb_med = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.1, reg_pred=1.e+4, reg_target=.2)
M_unb_big = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.1, reg_pred=1.e+4, reg_target=1.e+4)
M_balanced = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.1)


matches = [M_softmin_pred(input, target)[0, :, :],
           M_unb_small(input, target)[0, :, :],
           M_unb_med(input, target)[0, :, :],
           M_unb_big(input, target)[0, :, :],
           M_balanced(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['SoftMin from\nthe predictions',
                                                     'Unbalanced Sink.\n(low reg_target)',
                                                     'Unbalanced Sink.\n(medium reg_target)',
                                                     'Unbalanced Sink.\n(high reg_target)',
                                                     'Balanced\nSinkhorn'],
                                          title='Effect of reg_target (reg=0.1)',
                                          figsize=(20, 7))
fig_matches.show()
