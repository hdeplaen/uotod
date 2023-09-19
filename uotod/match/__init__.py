from .Hungarian import Hungarian
from .Min import Min
# from .SoftMin import SoftMin
from .BalancedSinkhorn import BalancedSinkhorn
from .UnbalancedSinkhorn import UnbalancedSinkhorn
from .BalancedPOT import BalancedPOT
from .UnbalancedPOT import UnbalancedPOT



# def _advise_model(self, reg, reg_pred, reg_gt, current_class):
#     EQUIV_INF_REG = 1.e+3
#     EQUIV_ZERO_REG = 1.e-3
#
#     if reg >= EQUIV_INF_REG and reg_pred >= EQUIV_INF_REG and reg_gt >= EQUIV_INF_REG:
#         advise = Hungarian
#         extra = ""
#     elif reg < EQUIV_INF_REG and reg_pred >= EQUIV_INF_REG and reg_gt >= EQUIV_INF_REG:
#         advise = _BalancedSinkhorn
#         extra =
#     elif reg > EQUIV_INF_REG and reg_pred < EQUIV_ZERO_REG and reg_gt > EQUIV_INF_REG:
#         advise = Min