from typing import Dict, List, Union, Optional, Tuple
from hidet.ir.node import Node
from hidet.ir.type import TypeNode, FuncType
from hidet.ir.expr import Var, Constant
from hidet.ir.stmt import Stmt


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
        'packed_func': Function
            the target function that this packed_func has packed. valid when attrs['kind'] == 'packed_func'
        'label': str
            the label of this function when it is in a function group
    """

    def __init__(self, name: str, params, body, ret_type, kind: str, local_vars, local_const_vars=None, extern_vars=None, attrs=None):
        self.name = name.replace('.', '_')
        self.kind = kind
        assert isinstance(kind, str) and kind in ['cuda_device', 'cuda_kernel', 'host_kernel', 'packed_func']
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: TypeNode = ret_type
        self.local_vars: List[Var] = local_vars
        self.local_const_vars: List[Tuple[Var, Constant]] = local_const_vars if local_const_vars else []
        self.extern_vars: List[Var] = extern_vars if extern_vars else []
        self.attrs = attrs if attrs else {}

        assert all(attr in self.valid_attrs for attr in self.attrs)

    def annotate(self, attr_name, attr_value, update=False):
        assert attr_name in self.valid_attrs
        if attr_name in self.attrs and not update:
            raise AttributeError(f'{attr_name} has existed')
        self.attrs[attr_name] = attr_value

    def get_attr(self, attr_name, default=None):
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        return default


class IRModule(Node):
    def __init__(self, funcs=None, task=None, global_vars=None):
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
                else:
                    raise ValueError('Function {} has already existed in module while include another module.'.format(name))
            else:
                self.functions[name] = func

        for name, var in module.global_vars.items():
            self.global_vars[name] = var

    def lookup(self, name_or_var: Union[str, Var]):
        if isinstance(name_or_var, Var):
            name = name_or_var.hint
        else:
            name = name_or_var
        if name not in self.functions:
            raise KeyError('Function {} does not exist in module, existed functions: \n{}.'.format(name, list(self.functions.keys())))
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

    def add(self, name, func: Function):
        if name in self.functions:
            raise ValueError('Function {} has already existed in module.'.format(name))
        else:
            self.functions[name] = func


