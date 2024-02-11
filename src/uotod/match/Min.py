from __future__ import annotations
import re

from .ClosestPrediction import ClosestPrediction
from .ClosestTarget import ClosestTarget
from ..utils.kwargs import kwargs_decorator


class Min:
    r"""
    This class only serves as a decorator for :class:`uotod.match.ClosestPrediction` and :class:`uotod.match.ClosestTarget`.
    ``uotod.match.Min(source='prediction', *args*, **kwargs)`` will return the same as ``uotod.match.ClosestTarget(*args, **kwargs)`` and
    ``uotod.match.Min(source='target', *args*, **kwargs)`` will return the same as ``uotod.match.ClosestPrediction(*args, **kwargs)``.
    This is meant to be consistent with the :class:`uotod.match.SoftMin` class.

    :param source: Either "target" (default) or "prediction".
    :type source: str, optional
    """
    @kwargs_decorator({"source": "target"})
    def __new__(cls, *args, **kwargs) -> ClosestTarget | ClosestPrediction:
        source = kwargs['source']
        assert isinstance(source, str), \
            TypeError("The property target is not set to a string.")
        source = re.sub(r'[^a-zA-Z]', '', source).lower()
        if source == "target" or source == "targets":
            return ClosestPrediction(*args, **kwargs)
        elif source == "prediction" or source == "predictions":
            return ClosestTarget(*args, **kwargs)
        else:
            raise NameError(f"The value for the closest can only be 'target' or 'prediction'.")
