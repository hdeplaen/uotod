import uotod
from uotod.sample import input, target

L = uotod.loss.GIoULoss()

H_hungarian = uotod.match.Hungarian(loc_match_cost=L, background_cost=0.8)
H_balanced_small_reg = uotod.match.BalancedSinkhorn(loc_match_cost=L, background_cost=0.8, reg=0.01)
H_balanced_big_reg = uotod.match.BalancedSinkhorn(loc_match_cost=L, background_cost=0.8, reg=0.1)

match_hungarian = H_hungarian(input, target)[0, :, :]
match_balanced_small_reg = H_balanced_small_reg(input, target)[0, :, :]
match_balanced_big_reg = H_balanced_big_reg(input, target)[0, :, :]

uotod.plot.match(match_hungarian, title='Hungarian (reg=0)')
uotod.plot.match(match_balanced_small_reg, title='Balanced Sinkhorn \n with reg=0.01')
uotod.plot.match(match_balanced_big_reg, title='Balanced Sinkhorn \n with reg=0.1')
