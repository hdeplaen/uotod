import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')

M_hungarian = uotod.match.Hungarian(loc_match_module=L, background_cost=0.8)
M_balanced_small_reg = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.01)
M_balanced_big_reg = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=0.1)

match_hungarian = M_hungarian(input, target)[0, :, :]
match_balanced_small_reg = M_balanced_small_reg(input, target)[0, :, :]
match_balanced_big_reg = M_balanced_big_reg(input, target)[0, :, :]

fig_img, fig_cost, _ = M_hungarian.plot(idx=0, img=imgs, plot_match=False)
fig_img.show()
fig_cost.show()

fig_matches = uotod.plot.multiple_matches([match_hungarian, match_balanced_small_reg, match_balanced_big_reg],
                                          subtitles=['Hungarian\nAlgorithm',
                                                     'Balanced Sinkhorn with\nsmall regularization',
                                                     'Balanced Sinkhorn with\nbig regularization'],
                                          title='Effect of the regularization',
                                          figsize=(12, 7))
fig_matches.show()
