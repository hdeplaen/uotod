import unittest
import itertools
import uotod
from uotod.sample import input, target


class TestMatch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMatch, self).__init__(*args, **kwargs)

    def test_hungarian(self):
        L = uotod.loss.GIoULoss()

        background_args = [True, False]

        for background_arg in background_args:
            H = uotod.match.Hungarian(loc_match_module=L,
                                      background=background_arg,
                                      bg_cost=0.8)
            H(input, target)

    def test_min(self):
        L = uotod.loss.GIoULoss()

        background_args = [True, False]
        source_args = ['target', 'prediction']
        threshold_args = [0., .5]

        for background_arg, source_arg, threshold_arg in itertools.product(background_args, source_args, threshold_args):
            H = uotod.match.Min(loc_match_module=L,
                                background=background_arg,
                                source=source_arg,
                                threshold=threshold_arg)
            H(input, target)

    def test_softmin(self):
        L = uotod.loss.GIoULoss()

        background_args = [True, False]

        for background_arg in background_args:
            H = uotod.match.SoftMin(loc_match_module=L,
                                      background=background_arg,
                                      bg_cost=0.8)
            H(input, target)

    def test_balanced(self):
        L = uotod.loss.GIoULoss()
        H = uotod.match.BalancedSinkhorn(loc_match_module=L, bg_cost=0.8)
        H(input, target)

    def test_unbalanced(self):
        L = uotod.loss.GIoULoss()
        H = uotod.match.UnbalancedSinkhorn(loc_match_module=L, bg_cost=0.8)
        H(input, target)
