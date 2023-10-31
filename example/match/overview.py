import uotod
from uotod.sample import input, target, imgs

# PARAMETERS
REG_LOW, REG_HIGH = 0.01, 0.08      # different scale of entropic regularization
BACKGROUND_COST = 0.8               # background cost
NUM_ITER = 100                      # number of iterations for Sinkhorn's algorithm (increased to ensure total convergence)
L = uotod.loss.GIoULoss(reduction='none')           # GIoU loss between the predictions and the targets (no classes taken into account)

M_00 = uotod.match.ClosestTarget(loc_match_module=L, background_cost=BACKGROUND_COST)
M_01 = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=BACKGROUND_COST, reg=REG_LOW, reg_target=0.2, reg_pred=50., num_iter=NUM_ITER)
M_02 = uotod.match.Hungarian(loc_match_module=L, background_cost=BACKGROUND_COST)
M_03 = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=REG_LOW, reg_target=100., reg_pred=0.1)
M_04 = uotod.match.ClosestPrediction(loc_match_module=L, background_cost=BACKGROUND_COST)

M_10 = uotod.match.SoftMin(loc_match_module=L, background_cost=BACKGROUND_COST, reg=REG_HIGH, source='prediction')
M_11 = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=BACKGROUND_COST, reg=REG_HIGH, reg_target=0.2, reg_pred=50., num_iter=NUM_ITER)
M_12 = uotod.match.BalancedSinkhorn(loc_match_module=L, background_cost=BACKGROUND_COST, reg=REG_HIGH, num_iter=NUM_ITER)
M_13 = uotod.match.UnbalancedSinkhorn(loc_match_module=L, background_cost=0.8, reg=REG_HIGH, reg_target=100., reg_pred=0.1)
M_14 = uotod.match.SoftMin(loc_match_module=L, background_cost=BACKGROUND_COST, reg=REG_HIGH, source='target')

m_00 = M_00(input, target)[0, :, :]
m_01 = M_01(input, target)[0, :, :]
m_02 = M_02(input, target)[0, :, :]
m_03 = M_03(input, target)[0, :, :]
m_04 = M_04(input, target)[0, :, :]

m_10 = M_10(input, target)[0, :, :]
m_11 = M_11(input, target)[0, :, :]
m_12 = M_12(input, target)[0, :, :]
m_13 = M_13(input, target)[0, :, :]
m_14 = M_14(input, target)[0, :, :]

fig = uotod.plot.multiple_matches([m_00, m_01, m_02, m_03, m_04,
                                   m_10, m_11, m_12, m_13, m_14],
                                  title="",
                                  subtitles=['Closest\nTarget',
                                             f"Unbalanced\nSinkhorn",
                                             "Hungarian\nAlgorithm",
                                             f"Unbalanced\nSinkhorn",
                                             'Closest\nPrediction',
                                             "SoftMin from\nthe predictions",
                                             f"Unbalanced\nSinkhorn",
                                             f"Balanced\nSinkhorn",
                                             f"Unbalanced\nSinkhorn",
                                             "SoftMin from\nthe targets"],
                                  subplots_disp=(2, 5),
                                  figsize=(14, 10))
fig.show()
