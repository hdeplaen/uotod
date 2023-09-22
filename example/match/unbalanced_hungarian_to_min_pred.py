import uotod
from uotod.sample import input, target, imgs

BACKGROUND_COST = 0.8
L = uotod.loss.GIoULoss()

H_min_pred = uotod.match.Min(loc_match_cost=L, background_cost=BACKGROUND_COST, source='prediction')
H_unbalanced = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, background_cost=BACKGROUND_COST, reg=0.01, reg_target=0.1, reg_pred=20.)
H_hungarian = uotod.match.Hungarian(loc_match_cost=L, background_cost=BACKGROUND_COST)

match_min_pred = H_min_pred(input, target)[0, :, :]
match_unbalanced = H_unbalanced(input, target)[0, :, :]
match_hungarian = H_hungarian(input, target)[0, :, :]

fig = uotod.plot.multiple_matches([match_min_pred, match_unbalanced, match_hungarian],
                            title='Unbalanced OT as interpolation between\n a minimum and the Hungarian algorithm.',
                            subtitles=['Minimim', 'Unbalanced Sinkhorn', 'Hungarian'])
fig.show()
a, b, _ = H_min_pred.plot(idx=0, img=imgs)
a.show()
b.show()