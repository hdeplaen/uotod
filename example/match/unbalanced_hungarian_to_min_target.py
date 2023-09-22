import uotod
from uotod.sample import input, target

L = uotod.loss.GIoULoss()

H_min_pred = uotod.match.Min(loc_match_cost=L, background_cost=0.8, closest='target')
H_unbalanced_pred = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, background_cost=0.8, reg=0.01, reg_target=100., reg_pred=0.1)
H_hungarian = uotod.match.Hungarian(loc_match_cost=L, background_cost=0.8)

match_min_pred = H_min_pred(input, target)[0, :, :]
match_unbalanced_pred_small_reg = H_unbalanced_pred(input, target)[0, :, :]
match_unbalanced_pred_big_reg = H_hungarian(input, target)[0, :, :]

uotod.plot.match(match_min_pred, title='Minimum')
uotod.plot.match(match_unbalanced_pred_small_reg, title='Unbalanced Sinkhorn')
uotod.plot.match(match_unbalanced_pred_big_reg, title='Hungarian')