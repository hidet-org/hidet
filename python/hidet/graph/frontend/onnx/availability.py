try:
    import onnx  # pylint: disable=unused-import
except ImportError:
    onnx = None
    _available = False
else:
    _available = True


def available():
    """
    Check if onnx is installed.

    Returns
    -------
    ret: bool
        True if torch is installed.
    """
    return _available
