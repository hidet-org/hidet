def set_module(module):
    """
    Set the module of the given class/function to the given module.
    """

    def decorator(obj):
        if module is not None:
            obj.__module__ = module
        return obj

    return decorator
