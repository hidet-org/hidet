from typing import Dict, List, Union, Optional
import string
from hidet.ir.node import Node
from hidet.ir.type import TypeNode, FuncType
from hidet.ir.expr import Var, Call
from hidet.ir.stmt import Stmt


def check_func_name(name: str):
    if len(name) == 0:
        raise ValueError('Do not allow empty function name.')
    for c in name:
        if not (c in string.ascii_lowercase or
                c in string.ascii_uppercase or
                c in string.digits or
                c in '_'):
            raise ValueError('Cannot use {} in function name'.format(repr(c)))


class Function(Node):
    valid_attrs = [
        'kind',
        'packed_func',
        'label',
        'kind',
        'cuda_grid_dim',
        'cuda_block_dim',
        'cuda_dynamic_smem_bytes',
        'cuda_min_blocks'
    ]
    """
    Valid Attrs:
        'kind': str, candidates: 'cuda_device', 'cuda_kernel', 'host_kernel', 'packed_func'
            the kind of this function.
                - 'cuda_device': this is a cuda device function, can only be called by cuda function
                - 'cuda_kernel': this is a cuda kernel function
                - 'host_kernel': this is a cpu kernel function
                - 'packed_func': this is a packed function that wraps kernel function(s)
        'cuda_grid_dim': Union[int, List[int]]
            the grid dimension in cuda launch configuration
        'cuda_block_dim': Union[int, List[int]]
            the block dimension in cuda launch configuration
        'cuda_dynamic_smem_bytes': int
            the dynamic shared memory in cuda launch configuration
        'cuda_min_blocks': int
            the minimal number of thread blocks in launch bound of cuda kernel function
        'packed_func': Var
            the var of target function that this packed_func has packed. valid when attrs['kind'] == 'packed_func'
        'label': str
            the label of this function when it is in a function group
    """

    def __init__(self, name: str, params, body, ret_type, kind: str, extern_vars=None, attrs=None):
        check_func_name(name)
        self.name: str = name
        self.kind = kind
        assert isinstance(kind, str) and kind in ['cuda_device', 'cuda_kernel', 'host_kernel', 'packed_func']
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: TypeNode = ret_type
        self.extern_vars: List[Var] = extern_vars if extern_vars else []
        self.attrs = attrs if attrs else {}

    def __call__(self, *args, **kwargs) -> Call:
        raise ValueError('Can only call script function in another script function, or lower it to execute.')

    def get_attr(self, attr_name, default=None):
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        return default


class IRModule(Node):
    def __init__(self, funcs=None, task=None, global_vars=None):
        # pylint: disable=import-outside-toplevel
        from hidet.ir.task import Task
        if funcs:
            assert isinstance(funcs, dict)
            # assert task is not None, 'Please specify the task'
        self.task: Optional[Task] = task
        self.functions: Dict[str, Function] = funcs if funcs else {}
        self.global_vars: Dict[str, Var] = global_vars if global_vars else {}

    def include(self, module, skip_duplicated=True):
        for name, func in module.functions.items():
            if name in self.functions:
                if skip_duplicated:
                    continue
                raise ValueError(f'Function {name} has already existed in module while include another module.')
            self.functions[name] = func

        for name, var in module.global_vars.items():
            self.global_vars[name] = var

    def lookup(self, name_or_var: Union[str, Var]):
        if isinstance(name_or_var, Var):
            name = name_or_var.hint
        else:
            name = name_or_var
        if name not in self.functions:
            raise ValueError('Function {} does not exist in module, existed functions: \n{}.'.format(
                name, list(self.functions.keys())))
        return self.functions[name]

    def lookup_var(self, name):
        assert name in self.functions, (name, self.functions.keys())
        if name not in self.global_vars:
            func = self.functions[name]
            if isinstance(func, Function):
                self.global_vars[name] = Var(name, FuncType.from_func(func))
            else:
                raise ValueError()

        return self.global_vars[name]

    def update_function(self, func: Function):
        self.functions[func.name] = func
        if func.name in self.global_vars:
            self.global_vars[func.name].type = func.name, FuncType.from_func(func)

    def add(self, name, func: Function):
        if name in self.functions:
            raise ValueError('Function {} has already existed in module.'.format(name))
        else:
            self.functions[name] = func
