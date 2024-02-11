import uotod
from uotod.sample import input, target, imgs


loc_module = uotod.loss.GIoULoss(reduction='none')
matching_method = uotod.match.BalancedSinkhorn(loc_match_module=loc_module, background_cost=0.8, compiled=False)
matching_method.to("cuda")

output = matching_method(input, target)

print(output)

fig_img, fig_cost, fig_match = matching_method.plot(idx=0, img=imgs, max_background_match=1., background=True)
fig_img.show()
fig_cost.show()
fig_match.show()