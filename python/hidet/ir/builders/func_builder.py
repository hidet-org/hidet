from typing import List, Dict

from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.stmt import Stmt
from hidet.ir.task import Grid, ThreadBlock, Warp, Thread


class FunctionBuilder:
    def __init__(self, name: str, ret_type=VoidType(), attrs=None):
        self.name = name
        self.params: List[Var] = []
        self.ret_type = ret_type
        self.local_vars = []
        self.func: Function = None
        self.body: Stmt = None
        self.extern_vars = {}
        self.attrs: Dict[str] = attrs if attrs else {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish()

    def extend_params(self, params: List[Var]):
        self.params.extend(params)

    def extend_extern_vars(self, extern_vars: List[Var]):
        self.extern_vars.update({var.name: var for var in extern_vars})

    def extend_local_vars(self, local_vars: List[Var]):
        self.local_vars.extend(local_vars)

    def extend_attrs(self, new_attrs: Dict[str, object]):
        self.attrs.update(new_attrs)

    def set_body(self, body: Stmt):
        self.body = body

    def finish(self):
        from hidet.ir.primitives import block_idx, thread_idx
        assert self.func is None
        if isinstance(self.attrs['worker'], (Grid, ThreadBlock, Warp, Thread)):
            self.extern_vars = [block_idx(), thread_idx()]
        self.func = Function(self.name, self.params, self.body, self.ret_type, self.local_vars, self.extern_vars, self.attrs)

    def get(self) -> Function:
        assert self.func.body is not None
        return self.func
