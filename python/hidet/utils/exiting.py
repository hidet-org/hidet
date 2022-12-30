"""
Detecting if the python interpreter is exiting.

This module provides a function to detect if the python interpreter is exiting. When the interpreter is exiting, some
resources (e.g., hidet.runtime.MemoryPool, hidet.cuda.Stream, hidet.cuda.Graph, ...) would be freed via __del__ methods.
However, the __del__ methods also rely on some functions/modules (e.g., cuda.cudart, ...) which may already have been
freed before the interpreter calls __del__ methods. In that case, the __del__ methods should do nothing but return. This
module provides a function to detect if the interpreter is in the exiting status via :func:`is_exiting`. We can use
this function to differentiate the normal __del__ call when on object is recycled by the garbage collector and the
interpreter-exiting __del__ call.

See Also
--------
Similar issue in cupy: https://github.com/cupy/cupy/pull/2809
"""
import atexit

_is_exiting = False


def _at_exit():
    global _is_exiting
    _is_exiting = True


atexit.register(_at_exit)


def is_exiting():
    """Returns True if the python interpreter is exiting."""
    return _is_exiting
