import uotod
from sample import input, target, img


loc_module = uotod.loss.GIoULoss()
# matching_method = uotod.match.BalancedPOT(method='sinkhorn', loc_matching_module=L, bg_cost=0.8, compiled=False)
matching_method = uotod.match.BalancedSinkhorn(loc_matching_module=loc_module, bg_cost=0.8, compiled=False)
matching_method.to("cuda")

output = matching_method(input, target)

print(output)

fig_img, fig_cost, fig_match = matching_method.plot(img=img, max_background_match=1., background=True)
fig_img.show()
fig_cost.show()
fig_match.show()