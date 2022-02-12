from typing import List, Dict

from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.stmt import Stmt


class FunctionBuilder:
    def __init__(self, name: str, ret_type=VoidType(), attrs=None):
        self.name = name
        self.params: List[Var] = []
        self.ret_type = ret_type
        self.local_vars = []
        self.func: Function = None
        self.body: Stmt = None
        self.attrs: Dict[str] = attrs if attrs else {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def extend_params(self, params: List[Var]):
        self.params.extend(params)

    def extend_local_vars(self, local_vars: List[Var]):
        self.local_vars.extend(local_vars)

    def extend_attrs(self, new_attrs: Dict[str, object]):
        self.attrs.update(new_attrs)

    def set_body(self, body: Stmt):
        self.body = body

    def finish(self):
        assert self.func is None
        self.func = Function(self.name, self.params, self.body, self.ret_type, self.local_vars, self.attrs)

    def get(self) -> Function:
        assert self.func.body is not None
        return self.func
