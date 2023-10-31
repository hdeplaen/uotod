import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')
M1 = uotod.match.ClosestPrediction()
M2 = uotod.match.ClosestPrediction()
M = uotod.match.WeightedSum(loc_match_module=L, background_cost=.8, matching_modules=[M1, M2])

M(input, target)

fig_img, fig_cost, fig_match = M.plot(idx=0, img=imgs)
_, fig_matches = M.plots_individual(idx=0)

fig_img.show()
fig_cost.show()
fig_match.show()
fig_matches.show()