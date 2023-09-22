import uotod
from uotod.sample import input, target

L = uotod.loss.GIoULoss()

H_unb_weak_constraints = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, background_cost=0.8, reg_pred=.1, reg_target=.1)
H_unb_strong_constraints = uotod.match.UnbalancedSinkhorn(loc_match_cost=L, background_cost=0.8, reg_pred=.5, reg_target=.5)
H_balanced = uotod.match.BalancedSinkhorn(loc_match_cost=L, background_cost=0.8)


match_unb_weak_constraints = H_unb_weak_constraints(input, target)[0, :, :]
match_unb_strong_constraints = H_unb_strong_constraints(input, target)[0, :, :]
match_balanced = H_balanced(input, target)[0, :, :]

uotod.plot.match(match_unb_weak_constraints, title='Unbalanced Sinkhorn \n (reg_pred=reg_target=0.1)')
uotod.plot.match(match_unb_strong_constraints, title='Unbalanced Sinkhorn \n (reg_pred=reg_target=0.5)')
uotod.plot.match(match_balanced, title='Balanced Sinkhorn \n (hard constraints)')
