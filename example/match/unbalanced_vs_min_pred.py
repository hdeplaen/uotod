import uotod
from sample import input, target

L = uotod.loss.GIoULoss()

H_min_pred = uotod.match.Min(loc_match_cost=L, bg_cost=0.8, source='prediction')
H_unbalanced_pred_small_reg = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, bg_cost=0.8, reg=0.01, reg_target=0.1, reg_pred=1.e+3)
H_unbalanced_pred_big_reg = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, bg_cost=0.8, reg=0.1, reg_target=0.1, reg_pred=1.e+3)

match_min_pred = H_min_pred(input, target)[0, :, :]
match_unbalanced_pred_small_reg = H_unbalanced_pred_small_reg(input, target)[0, :, :]
match_unbalanced_pred_big_reg = H_unbalanced_pred_big_reg(input, target)[0, :, :]

uotod.plot.match(match_min_pred, title='Minimum (reg=0)')
uotod.plot.match(match_unbalanced_pred_small_reg, title='Unbalanced Sinkhorn \n with reg=0.01')
uotod.plot.match(match_unbalanced_pred_big_reg, title='Unbalanced Sinkhorn \n with reg=0.1')