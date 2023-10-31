import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_softmin_high_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, source='prediction')
M_softmin_med_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.1, source='prediction')
M_softmin_low_reg = uotod.match.SoftMin(loc_match_module=L, background_cost=0.8, reg=0.01, source='prediction')
M_closest = uotod.match.ClosestTarget(loc_match_module=L, background_cost=0.8)

matches = [M_softmin_high_reg(input, target)[0, :, :],
           M_softmin_med_reg(input, target)[0, :, :],
           M_softmin_low_reg(input, target)[0, :, :],
           M_closest(input, target)[0, :, :]]

fig_matches = uotod.plot.multiple_matches(matches=matches,
                                          subtitles=['SoftMin from the\npredictions (reg=1, default)',
                                                     'SoftMin from the\npredictions (reg=0.1)',
                                                     'SoftMin from the\npredictions (reg=0.01)',
                                                     'Closest Target\n(minimum over the preds)'],
                                          title='Effect of the SoftMin regularization',
                                          figsize=(15, 7))
fig_matches.show()
