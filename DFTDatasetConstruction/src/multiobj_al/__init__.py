"""Top-level package for multi-objective active learning tools."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

__author__ = "Raphael Espinosa Casta\u00f1eda, Kangming Li, Daniel Persaud, Ashley Dale"

try:
    __version__ = version("multiobj-al")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "alignn_wrapper",
    "distill",
    "myfunc",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
