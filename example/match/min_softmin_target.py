import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_softmin_high_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, source='target')
M_softmin_med_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.1, source='target')
M_softmin_low_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.01, source='target')
M_min_target = uotod.match.Min(loc_match_module=L, background_cost=0.8, source='target', unmatched_to_background=False)
M_min_target2 = uotod.match.Min(loc_match_module=L, background_cost=0.8, source='target')

matches = [M_softmin_high_reg(input, target)[0, :, :],
           M_softmin_med_reg(input, target)[0, :, :],
           M_softmin_low_reg(input, target)[0, :, :],
           M_min_target(input, target)[0, :, :],
           M_min_target2(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['SoftMin from the\npredictions (reg=1, default)',
                                                     'SoftMin from the\npredictions (reg=0.1)',
                                                     'SoftMin from the\npredictions (reg=0.01)',
                                                     'Minimum from the predictions\n(unmatched_to_background=False)',
                                                     'Minimum from the\npredictions'],
                                          title='Effect of the SoftMin regularization',
                                          figsize=(20, 7))
fig_matches.show()
