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
from __future__ import annotations
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


class ForStmtAttr:
    def __init__(self, unroll=False, unroll_factor=None, unroll_explicit=False, parallel=False, parallel_threads=None):
        self.unroll: bool = unroll
        self.unroll_factor: Optional[int] = unroll_factor
        self.unroll_explicit: bool = unroll_explicit
        self.parallel: bool = parallel
        self.parallel_threads: Optional[int] = parallel_threads

    def __str__(self):
        if self.unroll is None:
            return '.'
        elif isinstance(self.unroll, bool):
            if self.unroll_explicit:
                return 'u+'
            else:
                return 'u'
        else:
            if self.unroll_explicit:
                return f'u{self.unroll}+'
            else:
                return f'u{self.unroll}'

    @staticmethod
    def parse(attr: str) -> List[ForStmtAttr]:
        """
        Parse the attribute string and return a list of ForStmtAttr.

        attr-string: attr+
        attr:
             | unroll
             | parallel
             | default
        unroll:
             | 'u'          # unroll
             | 'u' INT+     # unroll with factor, e.g., u1 u2 u3. u1 indicates unroll with factor 1 (i.e., no unroll)
             | 'u' '+'      # explicit unroll, will be unrolled by hidet instead of underlying compiler
        parallel:
             | 'p'          # parallel with available number of threads
             | 'p' INT+     # parallel with specified number of threads
        default: '.'


        Parameters
        ----------
        attr: str
            The attribute string.

        Returns
        -------
        attrs: List[ForStmtAttr]
            The list of ForStmtAttr.
        """
        s = attr.replace(' ', '')
        idx = 0

        def cur() -> Optional[str]:
            if idx >= len(s):
                return None
            return s[idx]

        attrs: List[ForStmtAttr] = []
        while idx < len(s):
            if s[idx] == '.':
                idx += 1
                attrs.append(ForStmtAttr())
            elif s[idx] == 'u':
                idx += 1
                c = cur()
                if c == '+':
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=True))
                    idx += 1
                elif c and c.isdigit():
                    unroll_factor = 0
                    while c and c.isdigit():
                        unroll_factor = unroll_factor * 10 + int(c)
                        idx += 1
                        c = cur()
                    if unroll_factor == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(unroll=True, unroll_factor=unroll_factor))
                else:
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=False))
            elif s[idx] == 'p':
                idx += 1
                c = cur()
                if c and c.isdigit():
                    parallel_threads = 0
                    while c and c.isdigit():
                        parallel_threads = parallel_threads * 10 + int(c)
                        idx += 1
                        c = cur()
                    if parallel_threads == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(parallel=True, parallel_threads=parallel_threads))
                else:
                    attrs.append(ForStmtAttr(parallel=True))
            else:
                raise ValueError(f"Invalid attribute string: {attr}")
        return attrs


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

    def __init__(self, loop_var, extent, body=None, *, attr: Optional[ForStmtAttr] = None):
        from hidet.ir.tools import simplify  # pylint: disable=import-outside-toplevel

        super().__init__()
        self.loop_var: Var = loop_var
        self.extent: Expr = simplify(convert(extent))
        self.body: Optional[Stmt] = body
        self.attr: ForStmtAttr = attr if attr else ForStmtAttr()


class ForMappingStmt(Stmt):
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
    is_volatile=False,
):
    from hidet.ir.tools import infer_type  # pylint: disable=import-outside-toplevel

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


Int = Union[Expr, int]


def launch_kernel(
    func_var: Var,
    args: Sequence[Expr],
    grid_dim: Union[Sequence[Int], Int],
    block_dim: Union[Sequence[Int], Int],
    shared_mem: Optional[Int] = 0,
) -> LaunchKernelStmt:
    launch_config: List[Tuple[Expr, Expr, Expr]] = []
    for dims in [grid_dim, block_dim]:
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        dims = list(dims)
        if len(dims) > 3:
            raise ValueError('Grid/Block dimension must be 3 or less.')
        while len(dims) < 3:
            dims.append(1)
        launch_config.append(convert(dims))
    grid_dim, block_dim = launch_config
    return LaunchKernelStmt(func_var, args, grid_dim, block_dim, convert(shared_mem))
