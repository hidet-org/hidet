from copy import copy
from typing import Dict, List
from hidet.ir.type import Type, FuncType
from hidet.ir.expr import Var
from hidet.ir.stmt import Stmt


class IRModule:
    def __init__(self, funcs):
        self.functions: Dict[str, Function] = funcs
        self.global_vars: Dict[str, Var] = {}

    def include(self, module):
        for name, func in module.functions.items():
            assert name not in self.functions
            self.functions[name] = func

        for name, var in module.global_vars.items():
            self.global_vars[name] = var

    def lookup(self, name):
        return self.functions[name]

    def lookup_var(self, name):
        assert name in self.functions
        if name not in self.global_vars:
            func = self.functions[name]
            self.global_vars[name] = Var(name, FuncType.from_func(func))
        return self.global_vars[name]


class Function:
    valid_attrs = [
        'worker',  'symbol_name'
    ]
    """
    Valid Attrs:
     'worker': one of 'host', 'grid', 'threadblock', 'warp', 'thread'
        the worker to run the function.
     'symbol_name': str
        the symbol name used in the source. if not defined, the func name is used.
    """

    def __init__(self, name, params, body, ret_type, local_vars, attrs=None):
        self.name = name
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: Type = ret_type
        self.local_vars: List[Var] = local_vars
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
        if default:
            return default
        raise AttributeError(f'{attr_name} not found')


