from typing import Dict, Union, Optional, List

from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.type import FuncType
from hidet.utils import green


class PrimitiveFunctionRegistry:
    def __init__(self, name: str, codegen_name: str, func_type: FuncType, function: Optional[Function] = None, generic: bool = False):
        self.var = Var(hint=name, type=func_type)
        self.name: str = name
        self.codegen_name: str = codegen_name
        self.func_type: FuncType = func_type
        self.function: Optional[Function] = function

        self.generic: bool = generic
        self.dispatch_dtype_rules: Dict[str, str] = {}

    def dispatch_dtype(self, dtype: str, dispatched_func_name: str):
        if not self.generic:
            raise ValueError('Can only dispatch a generic function.')
        if not is_primitive_function(dispatched_func_name):
            raise ValueError('Dispatched function {} does not exist during dispatching {} for data type {}.'.format(
                green(dispatched_func_name), green(self.name), green(dtype)
            ))
        self.dispatch_dtype_rules[dtype] = dispatched_func_name


class PrimitiveFunctionPool:
    def __init__(self):
        self.name2func: Dict[str, PrimitiveFunctionRegistry] = {}

    def register(self, name: str, func_or_type: Union[Function, FuncType], codegen_name: Optional[str], generic: bool):
        if isinstance(func_or_type, Function):
            if func_or_type.name != name:
                raise ValueError('The function name must be consistent, got {} and {}.'.format(name, func_or_type.name))
            if codegen_name is not None:
                if codegen_name != name:
                    raise ValueError('The codegen_name must be consistent, got {} and {}'.format(name, codegen_name))
                codegen_name = name
            registry = PrimitiveFunctionRegistry(
                name=name,
                codegen_name=codegen_name,
                func_type=FuncType.from_func(func_or_type),
                function=func_or_type,
                generic=generic
            )
        elif isinstance(func_or_type, FuncType):
            if codegen_name is None:
                codegen_name = name
            registry = PrimitiveFunctionRegistry(
                name=name,
                codegen_name=codegen_name,
                func_type=func_or_type,
                function=None,
                generic=generic
            )
        else:
            raise TypeError('Expect a Function or FuncType to register a primitive function, got {}'.format(type(func_or_type)))
        if name in self.name2func:
            raise KeyError('Primitive function {} has already registered.'.format(name))
        self.name2func[name] = registry
        return registry

    def lookup(self, func_var: Var) -> PrimitiveFunctionRegistry:
        if func_var.hint not in self.name2func:
            raise KeyError('Can not find primitive function via variable: {}.'.format(func_var))
        return self.name2func.get(func_var.hint)

    def lookup_by_name(self, name: str) -> PrimitiveFunctionRegistry:
        if name not in self.name2func:
            raise ValueError('Can not find primitive function with key: {}, candidates:\n{}.'.format(name, '\n'.join(str(v) for v in primitive_func_pool.name2func)))
        return self.name2func[name]

    def registered_names(self) -> Dict[str, List[str]]:
        ret = {}
        for name in self.name2func:
            target, func_name = name.split('.')
            if target not in ret:
                ret[target] = []
            ret[target].append(func_name)
        return ret

    def has_registered(self, key: str) -> bool:
        return key in self.name2func


primitive_func_pool = PrimitiveFunctionPool()


def is_primitive_function(name: str):
    return name in primitive_func_pool.name2func


def lookup_primitive_function(name: str) -> PrimitiveFunctionRegistry:
    return primitive_func_pool.lookup_by_name(name)


def registered_primitive_functions() -> List[str]:
    return list(primitive_func_pool.name2func.keys())


def register_primitive_function(name: str, func_or_type: Union[Function, FuncType], codegen_name: Optional[str] = None, generic=False) -> PrimitiveFunctionRegistry:
    """
    Register a primitive function.

    Parameters
    ----------
    name: str
        The name of the primitive function.

    func_or_type: Union[Function, FuncType]
        Function definition or function type of the primitive function.
        When function type is given, this function is implemented by underlying language (e.g., cuda c).

    codegen_name: Optional[str]
        The name used in code generation. When None is given, the 'name' parameter will be used.

    generic: bool
        Whether this function is a generic function. A generic function will be lowered to a concrete primitive
        function according to the calling arguments' type.

    Returns
    -------
    ret: PrimitiveFunctionRegistry
        The entry of registered primitive function.
    """
    return primitive_func_pool.register(name, func_or_type, codegen_name, generic)

