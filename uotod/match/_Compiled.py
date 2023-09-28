from warnings import warn
from abc import ABCMeta, abstractmethod
import sys
import importlib.util
from torch import Tensor

from ._Sinkhorn import _Sinkhorn
from ..utils import kwargs_decorator, extend_docstring

@extend_docstring(_Sinkhorn)
class _Compiled(_Sinkhorn, metaclass=ABCMeta):
    r"""
    :param compiled: Indicates whether to use a compiled version of the algorithm or not. Defaults to False.
    :param num_iter: Fixed number of iterations in Sinkhorn's algorithm. Defaults to 50 (can easily be lowered without hindering the global convergence).
    :type compiled: bool, optional
    :type num_iter: int, optional
    """

    @kwargs_decorator({'compiled': False,
                       'num_iter': 20})
    def __init__(self, **kwargs):
        super(_Compiled, self).__init__(**kwargs)
        self.compiled = kwargs["compiled"]
        self.num_iter = kwargs["num_iter"]

    @property
    def compiled(self) -> bool:
        r"""
        Boolean indicating whether to use a compiled version of the algorithm or not.
        """
        return self._compiled

    @compiled.setter
    def compiled(self, val: bool):
        assert isinstance(val, bool), \
            TypeError("The compiled property must be set to a boolean.")
        self._compiled = val
        if self._compiled:
            compiled_module = self._check_compilation()
            self._matching_method = getattr(compiled_module, self._compiled_name)
        else:
            self._matching_method = self._sinkhorn_python

    @property
    def num_iter(self) -> int:
        r"""
        Number of iterations in Sinkhorn's algorithm
        """
        return self._num_iter

    @num_iter.setter
    def num_iter(self, val: int):
        val = int(val)
        assert val > 0, "The number of iterations property must be strictly positive."
        self._num_iter = val

    @staticmethod
    def _check_compilation():
        r"""
        Check if Python Optimal Transport is installed and return it.
        :return: the ot package
        """
        name = 'uotod.compiled'
        if name in sys.modules:
            return sys.modules[name]
        elif (spec := importlib.util.find_spec(name)) is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        else:
            raise ModuleNotFoundError(
                "The compilation failed. Please use the inline version by setting the property compiled to False.")

    @abstractmethod
    def _sinkhorn_python(self):
        pass

    @property
    @abstractmethod
    def _compiled_name(self) -> str:
        pass





