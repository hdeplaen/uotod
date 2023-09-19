import uotod
from sample import input, target, img

L = uotod.loss.GIoULoss()
H = uotod.match.Min(loc_match_cost=L, background=False, source='target', threshold=.5)
H(input, target)

fig_img, fig_cost, fig_match = H.plot(img=img)

fig_img.show()
fig_cost.show()
fig_match.show()