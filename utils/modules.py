import sys
from typing import Optional, TypeVar

M = TypeVar("M")


def load_module(module_name: str, file_path: str) -> Optional[M]:
    """
    Load a module by name and search path

    Parameters
    ----------
    module_name : str
        Module name.
    file_path : str
        File path. It includes '.py'.

    Returns
    -------
    Optional[F]
        Returns `None` if Module could not be loaded.

    Examples
    --------
    >>> load_module('ref_local_tracking_model_003_2', 'ref_local_tracking/models/ref_local_tracking_model_003_2.py')
    """
    if sys.version_info >= (3, 5,):
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
    else:
        import imp

        mod = imp.load_source(module_name, file_path)
        return mod

