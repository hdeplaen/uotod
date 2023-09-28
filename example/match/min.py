import uotod
from uotod.sample import input, target, imgs

# PARAMETERS
IDX = 0
THRESHOLD = 0.5
L = uotod.loss.GIoULoss(reduction='none')

# DEFINE THE MATCHING STRATEGIES
M_pred = uotod.match.Min(loc_match_module=L, background=True, source='prediction', background_cost=.8)
M_pred_threshold = uotod.match.Min(loc_match_module=L, background=True, source='prediction', threshold=THRESHOLD, background_cost=.8)
M_target = uotod.match.Min(loc_match_module=L, background=True, source='target', background_cost=.8)
M_target_threshold = uotod.match.Min(loc_match_module=L, background=True, source='target', threshold=THRESHOLD, background_cost=.8)

# COMPUTE MATCHES
m_pred = M_pred(input, target)[IDX, :, :]
m_pred_threshold = M_pred_threshold(input, target)[IDX, :, :]
m_target = M_target(input, target)[IDX, :, :]
m_target_threshold = M_target_threshold(input, target)[IDX, :, :]

## ILLUSTRATIONS
fig_img, fig_cost, _ = M_pred.plot(idx=IDX, img=imgs, plot_match=False)
fig_img.show()
fig_cost.show()

fig_matches = uotod.plot.multiple_matches([m_pred, m_pred_threshold,
                                           m_target, m_target_threshold],
                                          subtitles=['From the\npredictions',
                                                     'From the predictions\nwith threshold',
                                                     'From the\ntargets',
                                                     'From the targets\nwith threshold'],
                                          subplots_disp=(2, 2),
                                          figsize=(8, 10))
fig_matches.show()
