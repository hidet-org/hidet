import sys
from importlib.util import find_spec
from importlib.metadata import version
from packaging.version import parse


def available():
    """
    Check if torch is installed.

    Returns
    -------
    ret: bool
        True if torch is installed.
    """
    spec = find_spec('torch')
    return spec is not None


def dynamo_available():
    """
    Check if torch is installed and torch dynamo is available.

    Returns
    -------
    ret: bool
        True if torch is installed and torch dynamo is available.
    """
    return available() and parse(version('torch')) >= parse('2.0.0.dev')


def imported():
    """
    Check if torch is imported.

    Returns
    -------
    ret: bool
        True if torch is imported.
    """
    return 'torch' in sys.modules
