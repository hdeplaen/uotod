from ._POT import _POT

class UnbalancedPOT(_POT):
    available_methods = ['sinkhorn',
                         'sinkhorn_stabilized',
                         'sinkhorn_reg_scaling']

    def __init__(self,
                 method: str,
                 **kwargs) -> None:
        assert method in UnbalancedPOT.available_methods, \
            f"Only the following methods are available in the balanced case: {BalancedPOT.available_methods}"
        super().__init__(method, balanced=True, **kwargs)