import uotod
from uotod.sample import input, target

L = uotod.loss.GIoULoss(reduction='none')

M_min = uotod.match.Min(loc_match_module=L, background_cost=0.8, source='target', unmatched_to_background=False)
M_softmin = uotod.match.SoftMin(loc_match_module=L, reg=0.01, background_cost=0.8, source='target')
M_unb_0 = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01, reg_pred=0, reg_target=1.e+4)
M_unb_small = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01, reg_pred=2.e-2, reg_target=1.e+4)
M_min2 = uotod.match.Min(loc_match_module=L, background_cost=0.8, source='target', unmatched_to_background=True)


matches = [M_min(input, target)[0, :, :],
           M_softmin(input, target)[0, :, :],
           M_unb_0(input, target)[0, :, :],
           M_unb_small(input, target)[0, :, :],
           M_min2(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['Minimum from the targets\n(unmatched_to_background=False)',
                                                     'SoftMin from the targets\n(reg=0.01)',
                                                     'Unbalanced Sinkhorn\n(reg_pred=0)',
                                                     'Unbalanced Sinkhorn\n(low reg_pred)',
                                                     'Minimum from the targets\n(unmatched_to_background=True)'],
                                          title='Influence of the unmatched_to_background argument',
                                          figsize=(20, 5))
fig_matches.show()