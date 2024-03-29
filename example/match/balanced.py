import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss(reduction='none')
H = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=0.8)
H(input, target)

fig_img, fig_cost, fig_match = H.plot(idx=0, img=imgs)
fig_img.show()
fig_cost.show()
fig_match.show()
