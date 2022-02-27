from . import func, vars

def is_reserved_name(name: str) -> bool:
    # noinspection PyProtectedMember
    return name in func._primitive_functions or name in vars._primitive_variables
