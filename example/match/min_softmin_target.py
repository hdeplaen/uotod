import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_softmin_high_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, source='target')
M_softmin_med_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.1, source='target')
M_softmin_low_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.01, source='target')
M_min_target = uotod.match.ClosestPrediction(loc_match_module=L, background_cost=0.8, unmatched_to_background=True)
M_min_target2 = uotod.match.ClosestPrediction(loc_match_module=L, background_cost=0.8)

matches = [M_softmin_high_reg(input, target)[0, :, :],
           M_softmin_med_reg(input, target)[0, :, :],
           M_softmin_low_reg(input, target)[0, :, :],
           M_min_target(input, target)[0, :, :],
           M_min_target2(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['SoftMin from the\ntargets (reg=1, default)',
                                                     'SoftMin from the\ntargets (reg=0.1)',
                                                     'SoftMin from the\ntargets (reg=0.01)',
                                                     'Closest Prediction\n(unmatched_to_background=True)',
                                                     'Closest Prediction\n(minimum over the targets)'],
                                          title='Effect of the SoftMin regularization',
                                          figsize=(20, 7))
fig_matches.show()
