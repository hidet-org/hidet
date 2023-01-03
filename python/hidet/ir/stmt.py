# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence, Tuple, Any, List, Union, Optional
import enum
from hidet.ir.node import Node
from hidet.ir.type import DataType, PointerType, TensorPointerType, ReferenceType
from hidet.ir.expr import Var, Expr, convert, Constant
from hidet.ir.compute import TensorNode
from hidet.ir.mapping import TaskMapping


# scope
class DeclareScope(enum.Enum):
    """
    The scope of a tensor variable used in declaration statement.
    """

    Default = 0
    Global = 1
    Shared = 2
    Register = 3

    @staticmethod
    def from_str(name):
        if name == 'global':
            return DeclareScope.Global
        elif name == 'shared':
            return DeclareScope.Shared
        elif name == 'register':
            return DeclareScope.Register
        else:
            return DeclareScope.Default


class Stmt(Node):
    pass


class EvaluateStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr: Expr = convert(expr)


class DeclareStmt(Stmt):
    def __init__(self, var, init: Optional[Expr] = None, is_static=False, scope: Optional[DeclareScope] = None):
        super().__init__()
        self.var: Var = var
        self.init: Optional[Expr] = convert(init)
        self.is_static: bool = is_static
        self.scope: Optional[DeclareScope] = scope if scope else DeclareScope.Default


class BufferStoreStmt(Stmt):
    def __init__(self, buf, indices, value, protected=False):
        super().__init__()
        assert isinstance(indices, (list, tuple)), type(indices)
        self.buf: Union[Var, TensorNode] = buf
        self.indices = convert(indices)
        self.value = convert(value)
        self.protected = protected


class AssignStmt(Stmt):
    def __init__(self, var, value):
        super().__init__()
        self.var: Var = var
        self.value: Expr = convert(value)


class ReturnStmt(Stmt):
    def __init__(self, ret_value: Optional[Expr] = None):
        super().__init__()
        self.ret_value: Optional[Expr] = ret_value


class LetStmt(Stmt):
    def __init__(self, bind_vars, bind_values, body=None):
        if not isinstance(bind_vars, (list, tuple)):
            bind_vars = [bind_vars]
        if not isinstance(bind_values, (list, tuple)):
            bind_values = [bind_values]
        assert len(bind_vars) == len(bind_values)
        assert len(bind_vars) > 0
        bind_values = [convert(bind_value) for bind_value in bind_values]
        self.bind_vars: List[Var] = bind_vars
        self.bind_values: List[Expr] = bind_values
        self.body: Optional[Stmt] = body


class ForStmt(Stmt):
    DEFAULT_UNROLL_LIMIT = 32

    def __init__(self, loop_var, extent, unroll: Optional[Union[int, bool]] = None, body=None):
        from hidet.ir.functors import simplify  # pylint: disable=import-outside-toplevel

        super().__init__()
        self.loop_var: Var = loop_var
        self.extent: Expr = simplify(convert(extent))
        self.unroll: Optional[Union[int, bool]] = unroll
        self.body: Optional[Stmt] = body


class ForTaskStmt(Stmt):
    def __init__(self, loop_vars: Sequence[Var], mapping: TaskMapping, worker: Expr, body: Stmt):
        self.loop_vars: List[Var] = list(loop_vars)
        self.mapping: TaskMapping = mapping
        self.worker: Expr = worker
        self.body: Stmt = body


class WhileStmt(Stmt):
    def __init__(self, cond: Expr, body: Stmt):
        self.cond: Expr = cond
        self.body: Stmt = body


class BreakStmt(Stmt):
    pass


class ContinueStmt(Stmt):
    pass


class IfStmt(Stmt):
    def __init__(self, cond: Expr, then_body=None, else_body=None):
        super().__init__()
        self.cond: Expr = convert(cond)
        self.then_body: Optional[Stmt] = then_body
        self.else_body: Optional[Stmt] = else_body


class AssertStmt(Stmt):
    def __init__(self, cond: Union[Expr, bool], msg: Optional[str]):
        super().__init__()
        self.cond: Expr = convert(cond)
        self.msg: Optional[str] = msg


class AsmStmt(Stmt):
    def __init__(
        self,
        template_string: str = "",
        outputs: Sequence[Tuple[str, Expr]] = (),
        inputs: Sequence[Tuple[str, Expr]] = (),
        is_volatile=False,
    ):
        self.template_string = template_string
        self.output_labels = [pr[0] for pr in outputs]
        self.output_exprs = [pr[1] for pr in outputs]
        self.input_labels = [pr[0] for pr in inputs]
        self.input_exprs = [pr[1] for pr in inputs]
        self.is_volatile = is_volatile


class BlackBoxStmt(Stmt):
    def __init__(self, template_string: str, *exprs: Expr):
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


class LaunchKernelStmt(Stmt):
    def __init__(
        self,
        func_var: Expr,
        args: Sequence[Expr],
        grid_dim: Tuple[Expr, Expr, Expr],
        block_dim: Tuple[Expr, Expr, Expr],
        shared_mem: Expr,
    ):
        self.func_var: Expr = func_var
        self.args: List[Expr] = list(args)
        self.grid_dim: Tuple[Expr, Expr, Expr] = grid_dim
        self.block_dim: Tuple[Expr, Expr, Expr] = block_dim
        self.shared_mem_bytes: Expr = shared_mem


def asm(
    template: str,
    *,
    outputs: Sequence[Any] = (),
    output_inputs: Sequence[Any] = (),
    inputs: Sequence[Any] = (),
    is_volatile=False
):
    from hidet.ir.functors import infer_type  # pylint: disable=import-outside-toplevel

    updated_outputs = []
    updated_inputs = []

    def get_register_type(expr: Expr) -> str:
        expr = convert(expr)
        expr_type = infer_type(expr)

        if isinstance(expr_type, ReferenceType):
            expr_type = expr_type.base_type

        if isinstance(expr_type, DataType):
            if isinstance(expr, Constant):
                return 'n'
            else:
                dtype2reg = {
                    'float16': 'h',
                    'float32': 'f',
                    'bfloat16': 'h',
                    'float64': 'd',
                    'uint8': 'h',
                    'uint16': 'h',
                    'uint32': 'r',
                    'uint64': 'l',
                    'int8': 'h',
                    'int16': 'h',
                    'int32': 'r',
                    'int64': 'l',
                }
                if expr_type.name not in dtype2reg:
                    raise NotImplementedError('{}'.format(expr_type))
                return dtype2reg[expr_type.name]
        elif isinstance(expr_type, (PointerType, TensorPointerType)):
            return 'l'
        else:
            raise ValueError('Can not deal with type {} in asm code.'.format(expr_type))

    for output in outputs:
        constraint = '=' + get_register_type(output)
        updated_outputs.append((constraint, convert(output)))
    for output_input in output_inputs:
        constraint = '+' + get_register_type(output_input)
        updated_outputs.append((constraint, convert(output_input)))
    for x in inputs:
        constraint = get_register_type(x)
        updated_inputs.append((constraint, convert(x)))
    return AsmStmt(template, updated_outputs, updated_inputs, is_volatile)
