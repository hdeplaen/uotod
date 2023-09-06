import uotod
from sample import input, target, img

L = uotod.loss.GIoULoss()
H = uotod.match.BalancedPOT(method = 'sinkhorn', loc_matching_module=L, bg_cost=0.8, compiled=False)
H(input, target)

fig_img, fig_cost, fig_match = H.plot(img=img, max_background_match=1., background=True)
fig_img.show()
fig_cost.show()
fig_match.show()