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
from typing import Optional, List, Set, Dict, Union, Mapping, Sequence
import itertools
import operator
from collections import defaultdict

from hidet.ir.type import DataType
from hidet.ir.expr import Expr, Var, Add, Sub, Multiply, FloorDiv, Mod, Constant, Div
from hidet.ir.func import Function
from hidet.ir.functors import FuncStmtExprVisitor
from hidet.ir.stmt import Stmt, ForStmt, LetStmt


# from hidet.ir.task import Grid, ThreadBlock, Warp, Thread, Host


class BoundInfo:
    _max_num_candidates = 1024
    _max_compute_iters = 128 * 128

    def __init__(self, value=None, candidates=None, min_value=None, max_value=None):
        # three level of bound information:
        # 1. know its value
        # 2. know its candidates
        # 3. know it min_value and/or max_value
        # the specific one will hide the loose one, e.g., candidates are ignored when value is present.
        self.value: Optional[int] = None
        self.candidates: Optional[Set[int]] = None
        self.min_value: Optional[int] = None
        self.max_value: Optional[int] = None
        if value is not None:
            self.value = value
        elif candidates:
            if len(candidates) == 1:
                self.value = candidates[0]
            else:
                self.candidates = candidates
        elif min_value is not None and max_value is not None:
            if min_value == max_value:
                self.value = min_value
            elif max_value - min_value <= BoundInfo._max_num_candidates:
                self.candidates = set(range(min_value, max_value + 1))
            else:
                self.min_value = min_value
                self.max_value = max_value
        else:
            self.min_value = min_value
            self.max_value = max_value

    @staticmethod
    def combine(lhs: 'BoundInfo', rhs: 'BoundInfo', op) -> 'BoundInfo':
        # pylint: disable=too-many-branches
        if not lhs.has_determent_range() or not rhs.has_determent_range():
            return BoundInfo()
        if lhs.is_empty_set() or rhs.is_empty_set():
            return BoundInfo(candidates={})

        lhs_candidates = lhs.candidate_set()
        rhs_candidates = rhs.candidate_set()
        if (
            lhs_candidates
            and rhs_candidates
            and len(lhs_candidates) * len(rhs_candidates) <= BoundInfo._max_compute_iters
        ):
            candidates = set()
            for lv in lhs_candidates:
                for rv in rhs_candidates:
                    candidates.add(op(lv, rv))
            if len(candidates) == 1:
                return BoundInfo(value=candidates.pop())
            else:
                return BoundInfo(candidates=candidates)
        else:
            # fall back to use min/max value of lhs_candidates/rhs_candidates to infer
            lhs_candidates = [lhs.possible_min_value(), lhs.possible_max_value()]
            rhs_candidates = [rhs.possible_min_value(), rhs.possible_max_value()]
            if op in [operator.add, operator.sub, operator.mul]:
                candidates = [
                    op(a, b)
                    for a, b in itertools.product(
                        [min(lhs_candidates), max(lhs_candidates)], [min(rhs_candidates), max(rhs_candidates)]
                    )
                ]
                return BoundInfo(min_value=min(candidates), max_value=max(candidates))
            elif op is operator.floordiv:
                if all(v > 0 for v in rhs_candidates):
                    return BoundInfo(
                        min_value=min(lhs_candidates) // max(rhs_candidates),
                        max_value=max(lhs_candidates) // min(rhs_candidates),
                    )
                else:
                    return BoundInfo()
            elif op is operator.mod:
                if rhs.possible_max_value() is not None:
                    return BoundInfo(min_value=0, max_value=rhs.possible_max_value() - 1)
                else:
                    return BoundInfo()
            else:
                raise NotImplementedError()

    def candidate_set(self):
        if self.value is not None:
            return [self.value]
        elif self.candidates is not None:
            return self.candidates
        else:
            return None

    def is_empty_set(self):
        return self.candidates is not None and len(self.candidates) == 0

    def has_determent_range(self) -> bool:
        if self.value is not None:
            return True
        if self.candidates is not None:
            return True
        if self.min_value is not None and self.max_value is not None:
            return True
        return False

    def possible_max_value(self):
        if self.value is not None:
            return self.value
        elif self.candidates:
            return max(self.candidates)
        elif self.max_value is not None:
            return self.max_value
        else:
            return None

    def possible_min_value(self):
        if self.value is not None:
            return self.value
        elif self.candidates:
            return min(self.candidates)
        elif self.min_value is not None:
            return self.min_value
        else:
            return None

    def is_one(self):
        return self.value == 1

    def is_zero(self):
        return self.value == 0

    def __add__(self, other):
        return self.combine(self, other, operator.add)

    def __sub__(self, other):
        return self.combine(self, other, operator.sub)

    def __mul__(self, other):
        return self.combine(self, other, operator.mul)

    def __floordiv__(self, other):
        return self.combine(self, other, operator.floordiv)

    def __mod__(self, other):
        return self.combine(self, other, operator.mod)

    def __lt__(self, other):
        lhs_max = self.possible_max_value()
        rhs_min = other.possible_min_value()
        return lhs_max is not None and rhs_min is not None and lhs_max < rhs_min

    def __le__(self, other):
        lhs_max = self.possible_max_value()
        rhs_min = other.possible_min_value()
        return lhs_max is not None and rhs_min is not None and lhs_max <= rhs_min

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        elif self.candidates is not None:
            return str(len(self.candidates)) + ': ' + str(self.candidates)
        elif self.min_value or self.max_value:
            return f'[{self.min_value}:{self.max_value}]'
        else:
            return 'Any'


Int = Union[int, Expr]


def normalize_launch_dims(dims: Union[Int, Sequence[Int]]) -> List[Union[Expr, int]]:
    if isinstance(dims, (list, tuple)):
        dims = list(dims)
        while len(dims) < 3:
            dims = dims + [1]
    else:
        dims = [dims, dims, dims]
    ret = []
    for dim in dims:
        if isinstance(dim, int):
            ret.append(dim)
        elif isinstance(dim, Expr):
            from hidet.ir.functors import simplify  # pylint: disable=import-outside-toplevel

            simplified_dim = simplify(dim)
            if isinstance(simplified_dim, Constant):
                assert isinstance(simplified_dim.type, DataType) and simplified_dim.type.is_integer()
                ret.append(simplified_dim.value)
            else:
                ret.append(simplified_dim)
        else:
            raise ValueError(dim)
    return ret


class BoundAnalyzer(FuncStmtExprVisitor):
    # we only infer bound based on variables from LetStmt and ForStmt, and the constants.
    # so the local variable with AssignStmt is not used infer bound.
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        FloorDiv: operator.floordiv,
        Mod: operator.mod,
        Div: operator.floordiv,  # for the node with BoundInfo, we are sure they are integers
    }

    def __init__(self, var2bound: Dict[Expr, BoundInfo] = None):
        # please give the bound of external variable such as threadIdx.x using var2bound parameter
        super().__init__()
        self.bound: Dict[Expr, BoundInfo] = defaultdict(BoundInfo)
        if var2bound:
            self.bound.update(var2bound)

    def visit_Function(self, func: Function):
        # note: we use the vars in func.extern_vars instead of hidet.ir.primitives.thread_idx for multiprocessing
        extern_var_map = {var.name: var for var in func.extern_vars}
        if func.kind in ['cuda_kernel', 'cuda_device']:
            if 'cuda_block_dim' in func.attrs:
                block_dims = normalize_launch_dims(func.attrs['cuda_block_dim'])
                for block_dim, suffix in zip(block_dims, ['x', 'y', 'z']):
                    if isinstance(block_dim, int):
                        bound_info = BoundInfo(min_value=0, max_value=int(block_dim) - 1)
                        self.bound[extern_var_map['threadIdx.{}'.format(suffix)]] = bound_info
            if 'cuda_grid_dim' in func.attrs:
                grid_dims = normalize_launch_dims(func.attrs['cuda_grid_dim'])
                for grid_dim, suffix in zip(grid_dims, ['x', 'y', 'z']):
                    if isinstance(grid_dim, int):
                        bound_info = BoundInfo(min_value=0, max_value=int(grid_dim) - 1)
                        self.bound[extern_var_map['blockIdx.{}'.format(suffix)]] = bound_info
        self.visit(func.body)

    def combine(self, e: Union[Add, Sub, Multiply, FloorDiv, Mod, Div]):
        self.visit(e.a)
        self.visit(e.b)
        self.bound[e] = BoundAnalyzer.op_dict[e.__class__](self.bound[e.a], self.bound[e.b])

    def visit_Add(self, e: Add):
        self.combine(e)

    def visit_Sub(self, e: Sub):
        self.combine(e)

    def visit_Multiply(self, e: Multiply):
        self.combine(e)

    def visit_Div(self, e: Div):
        self.combine(e)

    def visit_FloorDiv(self, e: FloorDiv):
        self.combine(e)

    def visit_Mod(self, e: Mod):
        self.combine(e)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
            self.bound[bind_var] = self.bound[bind_value]
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit(stmt.extent)
        max_val = self.bound[stmt.extent].possible_max_value()
        if max_val is not None:
            max_val -= 1
        self.bound[stmt.loop_var] = BoundInfo(min_value=0, max_value=max_val)
        self.visit(stmt.body)

    def visit_Constant(self, e: Constant):
        if e.is_scalar() and e.type.name == 'int32':
            self.bound[e] = BoundInfo(value=e.value)


def infer_bound(
    node: Union[Function, Stmt, Expr], var2bound: Optional[Mapping[Var, BoundInfo]] = None
) -> Dict[Expr, BoundInfo]:
    visitor = BoundAnalyzer(var2bound)
    visitor.visit(node)
    return visitor.bound
