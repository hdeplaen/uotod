from abc import ABCMeta, abstractmethod


import sys
import importlib.util

from ..utils import kwargs_decorator


class _POT(metaclass=ABCMeta):
    r"""
    Makes you able to use special methods from the Python Optimal Transport Toolbox (not installed by default).
    """
    @kwargs_decorator({'method': 'sinkhorn',
                       'balanced': True})
    def __init__(self, **kwargs) -> None:
        super(_POT, self).__init__(**kwargs)
        ot_module = _POT._check_ot_installed()
        self._method_kwargs = kwargs
        if kwargs['balanced']:
            sub_module = ot_module.bregman
        else:
            sub_module = ot_module.unbalanced
        self._pot_method = getattr(sub_module, kwargs['method'])


    @staticmethod
    def _check_ot_installed():
        r"""
        Check if Python Optimal Transport is installed and return it.
        :return: the ot package
        """
        name = 'ot'
        if name in sys.modules:
            return sys.modules[name]
        elif (spec := importlib.util.find_spec(name)) is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        else:
            raise ModuleNotFoundError(
                r"""
                The Python Optimal Transport package must be installed in order to perform this matching.
                Please install the package by running
                    >> pip install pot
                (or via another way).
                """)