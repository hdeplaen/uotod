import uotod
from uotod.sample import input, target, imgs

L = uotod.loss.GIoULoss()
H = uotod.match.Min(loc_match_cost=L,
                    background=True,
                    source='target',
                    threshold=.5,
                    background_cost=.8)
H(input, target)

fig_img, fig_cost, fig_match = H.plot(idx=0, img=imgs)