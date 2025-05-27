import os

def validate_path(path, base):
    """
    Validate the path to prevent directory traversal attacks.

    This function checks if the path is inside the base directory.

    Parameters
    ----------
    path: str or List[str]
        The path to be validated.
    base: str
        The base directory to check against.

    Returns
    -------
    bool
        True if the path is valid, False otherwise.
    """
    if isinstance(path, list):
        # If path is a list, check each path in the list
        for p in path:
            if not validate_path(p, base):
                return False
        return True
    elif isinstance(path, str):
        # Normalize the paths
        path = os.path.realpath(path)
        base = os.path.realpath(base)

        # Check if the path is inside the base directory
        return os.path.commonpath([path, base]) == base
    else:
        raise TypeError('Invalid type for path: {}'.format(type(path)))
