from typing import Dict, Union, Optional, List

from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.type import FuncType


class PrimitiveFunctionRegistry:
    def __init__(self, space: str, name: str, func_type: FuncType, function: Optional[Function] = None, generic: bool = False):
        key = '{}.{}'.format(space, name)
        self.var = Var(hint=key, type=func_type)
        self.space: str = space
        self.name: str = name
        self.func_type: FuncType = func_type
        self.function: Optional[Function] = function

        self.generic = generic
        self.dispatch_dtype_rules: Dict[str, str] = {}

    def dispatch_dtype(self, dtype: str, space: str, func_name: str):
        if not self.generic:
            raise ValueError('Can only dispatch a generic function.')
        func_key = '{}.{}'.format(space, func_name)
        self.dispatch_dtype_rules[dtype] = func_key


class PrimitiveFunctionPool:
    def __init__(self):
        self.key2func: Dict[str, PrimitiveFunctionRegistry] = {}

    def register(self, space: str, name: str, func_or_type: Union[Function, FuncType], generic):
        if isinstance(func_or_type, Function):
            registry = PrimitiveFunctionRegistry(
                name=name,
                func_type=FuncType.from_func(func_or_type),
                space=space,
                function=func_or_type,
                generic=generic
            )
        elif isinstance(func_or_type, FuncType):
            registry = PrimitiveFunctionRegistry(
                name=name,
                func_type=func_or_type,
                space=space,
                function=None,
                generic=generic
            )
        else:
            raise TypeError('Expect a Function or FuncType to register a primitive function, got {}'.format(type(func_or_type)))
        key = '{}.{}'.format(space, name)
        if key in self.key2func:
            raise KeyError('Primitive function {} has already registered.'.format(key))
        self.key2func[key] = registry
        return registry

    def lookup(self, func_var: Var) -> PrimitiveFunctionRegistry:
        if func_var.hint not in self.key2func:
            raise KeyError('Can not find primitive function via variable: {}.'.format(func_var))
        return self.key2func.get(func_var.hint)

    def lookup_by_key(self, key: str) -> PrimitiveFunctionRegistry:
        if key not in self.key2func:
            raise KeyError('Can not find primitive function with key: {}.'.format(key))
        return self.key2func[key]

    def lookup_by_name(self, target: str, name: str) -> PrimitiveFunctionRegistry:
        key = '{}.{}'.format(target, name)
        if key not in self.key2func:
            candidates = '\n'.join(self.registered_names()[target])
            raise ValueError('Can not find primitive function with target "{}" and name "{}", candidates:\n{}'.format(target, name, candidates))
        return self.key2func[key]

    def registered_names(self) -> Dict[str, List[str]]:
        ret = {}
        for name in self.key2func:
            target, func_name = name.split('.')
            if target not in ret:
                ret[target] = []
            ret[target].append(func_name)
        return ret

    def has_registered(self, key: str) -> bool:
        return key in self.key2func


primitive_func_pool = PrimitiveFunctionPool()


def is_primitive_function(key: str):
    return key in primitive_func_pool.key2func


def lookup_primitive_function(key: str) -> PrimitiveFunctionRegistry:
    return primitive_func_pool.lookup_by_key(key)


def registered_primitive_functions() -> List[str]:
    return list(primitive_func_pool.key2func.keys())


def register_primitive_function(target, name, func_or_type: Union[Function, FuncType], generic=False) -> PrimitiveFunctionRegistry:
    """
    Register a primitive function.

    Parameters
    ----------
    target: str
        The target device of the primitive function works on. Candidates: 'base', 'cuda', 'cpu'.
        'base' indicates this function is generic to different devices.
        'cuda' indicates this is a primitive function in CUDA programming platform.
        'cpu' indicates this is a primitive function specific in CPU.

    name: str
        The name of the primitive function.

    func_or_type: Union[Function, FuncType]
        Function definition or function type of the primitive function.
        When function type is given, this function is implemented by underlying language (e.g., cuda c).

    generic: bool
        Whether this function is a generic function. A generic function will be lowered to a concrete primitive
        function according to the calling arguments' type.

    Returns
    -------
    ret: PrimitiveFunctionRegistry
        The entry of registered primitive function.
    """
    return primitive_func_pool.register(target, name, func_or_type, generic)

