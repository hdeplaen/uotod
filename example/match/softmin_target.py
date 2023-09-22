import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss()
H = uotod.match.SoftMin(loc_match_cost=L, background_cost=0.8, source='target')
H(input, target)

fig_img, fig_cost, fig_match = H.plot(idx=0, img=imgs)
