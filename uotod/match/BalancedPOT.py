from ._POT import _POT
from ._Sinkhorn import _Sinkhorn

class BalancedPOT(_POT, _Sinkhorn):
    available_methods = ['sinkhorn',
                         'greenkorn',
                         'screenkorn',
                         'sinkhorn_log',
                         'sinkhorn_stabilized']

    def __init__(self,
                 method: str,
                 **kwargs) -> None:
        assert method in BalancedPOT.available_methods, \
            f"Only the following methods are available in the balanced case: {BalancedPOT.available_methods}"
        super().__init__(method, balanced=True, **kwargs)