from typing import Sequence, Tuple
from typing import List, Union, Optional
from hidet.ir.node import Node
from hidet.ir.expr import Var, Expr, convert, Constant
from hidet.ir.mapping import TaskMapping


class Stmt(Node):
    pass


class EvaluateStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr = convert(expr)


class DeclareStmt(Stmt):
    def __init__(self, var, init: Optional[Expr] = None):
        super().__init__()
        self.var: Var = var
        self.init: Optional[Expr] = init


class BufferStoreStmt(Stmt):
    def __init__(self, buf, indices, value, protected=False):
        super().__init__()
        assert isinstance(indices, (list, tuple)), type(indices)
        self.buf = buf
        self.indices = convert(indices)
        self.value = convert(value)
        self.protected = protected


class AssignStmt(Stmt):
    def __init__(self, var, value):
        super().__init__()
        self.var = var
        self.value = convert(value)


class ReturnStmt(Stmt):
    def __init__(self, ret_value: Optional[Expr] = None):
        super().__init__()
        self.ret_value = ret_value


class LetStmt(Stmt):
    def __init__(self, bind_vars, bind_values, body=None):
        if not isinstance(bind_vars, (list, tuple)):
            bind_vars = [bind_vars]
        if not isinstance(bind_values, (list, tuple)):
            bind_values = [bind_values]
        assert len(bind_vars) == len(bind_values)
        assert len(bind_vars) > 0
        bind_values = [convert(bind_value) for bind_value in bind_values]
        self.bind_vars = bind_vars
        self.bind_values = bind_values
        self.body = body


class ForStmt(Stmt):
    DEFAULT_UNROLL_LIMIT = 32

    def __init__(self, loop_var, extent, unroll: Optional[Union[int, bool]] = None, body=None):
        from hidet.ir.functors import simplify
        super().__init__()
        self.loop_var: Var = loop_var
        self.extent = simplify(convert(extent))
        self.unroll = unroll
        self.body = body


class ForTaskStmt(Stmt):
    def __init__(self, loop_vars: Sequence[Var], mapping: TaskMapping, worker: Expr, body: Stmt):
        self.loop_vars: List[Var] = list(loop_vars)
        self.mapping: TaskMapping = mapping
        self.worker: Expr = worker
        self.body: Stmt = body


class IfStmt(Stmt):
    def __init__(self, cond: Expr, then_body=None, else_body=None):
        super().__init__()
        self.cond = convert(cond)
        self.then_body = then_body
        self.else_body = else_body


class AssertStmt(Stmt):
    def __init__(self, cond: Union[Expr, bool], msg: str):
        super().__init__()
        self.cond = convert(cond)
        self.msg = msg


class AsmStmt(Stmt):
    def __init__(self,
                 template_string: str = "",
                 outputs: Sequence[Tuple[str, Expr]] = (),
                 inputs: Sequence[Tuple[str, Expr]] = (),
                 is_volatile=False):
        self.template_string = template_string
        self.output_labels = [pr[0] for pr in outputs]
        self.output_exprs = [pr[1] for pr in outputs]
        self.input_labels = [pr[0] for pr in inputs]
        self.input_exprs = [pr[1] for pr in inputs]
        self.is_volatile = is_volatile


class BlackBoxStmt(Stmt):
    def __init__(self, template_string: str, *exprs: Sequence[Expr]):
        super().__init__()
        self.template_string: str = template_string
        self.exprs: Tuple[Expr] = convert(exprs)
        expect_args_num = self.template_string.count('{}')
        assert expect_args_num == len(exprs)


class SeqStmt(Stmt):
    def __init__(self, seq: List[Stmt]):
        super().__init__()
        self.seq: Tuple[Stmt] = tuple(seq)
        for stmt in seq:
            assert isinstance(stmt, Stmt), str(type(stmt))
