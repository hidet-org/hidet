from packaging import version

try:
    import torch
except ImportError:
    torch = None
    _available = False
else:
    _available = True

if not _available or version.parse(torch.__version__) < version.parse('2.0.0.dev'):
    _dynamo_available = False
else:
    _dynamo_available = True


def available():
    """
    Check if torch is installed.

    Returns
    -------
    ret: bool
        True if torch is installed.
    """
    return _available


def dynamo_available():
    """
    Check if torch is installed and torch dynamo is available.

    Returns
    -------
    ret: bool
        True if torch is installed and torch dynamo is available.
    """
    return _dynamo_available
