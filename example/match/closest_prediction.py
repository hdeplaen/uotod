import uotod
from uotod.sample import input, target, imgs

# PARAMETERS
IDX = 0
THRESHOLD = 0.5
L = uotod.loss.GIoULoss(reduction='none')

# DEFINE THE MATCHING STRATEGIES
M_target = uotod.match.ClosestPrediction(loc_match_module=L, background=True, background_cost=.8)
M_target_threshold = uotod.match.ClosestPrediction(loc_match_module=L, threshold=THRESHOLD, background_cost=.8)

# COMPUTE MATCHES
m_target = M_target(input, target)[IDX, :, :]
m_target_threshold = M_target_threshold(input, target)[IDX, :, :]

## ILLUSTRATIONS
fig_img, fig_cost, _ = M_target.plot(idx=IDX, img=imgs, plot_match=False)
fig_img.show()
fig_cost.show()

fig_matches = uotod.plot.multiple_matches([m_target, m_target_threshold],
                                          subtitles=['Closest predictions',
                                                     'Closest predictions\nwith threshold'],
                                          subplots_disp=(1, 2),
                                          figsize=(8, 5))
fig_matches.show()
