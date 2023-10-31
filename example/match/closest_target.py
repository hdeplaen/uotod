import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')
M_pred = uotod.match.ClosestTarget(loc_match_module=L, background=True, background_cost=.8)
M_pred(input, target)

fig_img, fig_cost, fig_match = M_pred.plot(idx=0, img=imgs)
fig_img.show()
fig_cost.show()
fig_match.show()
