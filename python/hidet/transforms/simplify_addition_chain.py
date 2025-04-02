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
from typing import Dict, Optional, Tuple, List, Callable

from hidet.ir import ForMappingStmt, AssignStmt, WhileStmt, Let, DeclareStmt, IRModule
from hidet.ir.expr import Add, Multiply, Mod, SymbolVar
from hidet.ir.expr import Div, Constant, Expr
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.primitives.cuda.vars import blockDim, gridDim, blockIdx, threadIdx
from hidet.ir.stmt import LetStmt, ForStmt
from hidet.ir.tools import TypeInfer, IRPrinter, collect
from hidet.transforms.base import FunctionPass


class DecomposeRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Multiply(self, e: Multiply):
        if isinstance(e.a, Constant) and isinstance(e.b, Constant):
            # c1 * c2 => c1*c2
            return Constant(value=e.a.value * e.b.value, const_type=self.type_infer(e))
        elif isinstance(e.a, Add) and isinstance(e.b, Constant):
            # (e1 + e2) * c => e1 * c + e2 * c
            e1, e2, c = e.a.a, e.a.b, e.b
            if self.type_infer(e1 + e2) == self.type_infer(e1 * c) == self.type_infer(e2 * c):
                return self.visit(e1 * c + e2 * c)
            else:
                return super().visit_Multiply(e)
        elif isinstance(e.a, Multiply) and isinstance(e.a.b, Constant) and isinstance(e.b, Constant):
            # (e1 * c1) * c2 => e1 * (c1 * c2)
            e1, c1, c2 = e.a.a, e.a.b, e.b
            if self.type_infer(e1 * c1) == self.type_infer(c1 * c2):
                # in case: (uint64 * uint32) * uint32 and c1 * c2 overflow
                return self.visit(e1 * (c1 * c2))
            else:
                return super().visit_Multiply(e)
        else:
            return super().visit_Multiply(e)


class DepthAnalyzer(IRVisitor):
    """
    Collect the information of a function, and use the collected information to generate the key used for
    sort the items in an addition chain. First need use this analyzer to visit the function, and then
    it can be used to generate the keys.
    """

    def __init__(self, global_vars, func, printer):
        super().__init__()
        self.var2depth: Dict[Var, int] = {}
        self.global_vars: list[Var] = global_vars
        self.func = func
        self.type_infer = TypeInfer()
        self.printer = printer
        self.current_depth = 1

        self.expr2depth: Dict[Expr, int] = {}

        self.printer(func)
        self.visit(func)

    def get_depth(self, expr: Expr):
        if expr in self.expr2depth:
            return self.expr2depth[expr]

        used_vars: List[Var] = collect(expr, node_types=(Var,))
        depth = 0
        for var in used_vars:
            if var.type.is_func_type():
                continue
            if var in self.var2depth:
                var_depth = self.var2depth[var]
            else:
                # used external variables
                continue
            depth = max(depth, var_depth)

        self.expr2depth[expr] = depth
        return depth

    def visit_Function(self, func: Function):
        global_invariants = [
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            blockIdx.x,
            blockIdx.y,
            blockIdx.z,
            threadIdx.x,
            threadIdx.y,
            threadIdx.z,
        ]
        for invariant in global_invariants:
            self.var2depth[invariant] = self.current_depth
        for global_var in self.global_vars:
            self.var2depth[global_var] = self.current_depth
        for param in func.params:
            self.var2depth[param] = self.current_depth
        self.current_depth += 1
        super().visit_Function(func)
        self.current_depth -= 1

    def visit_ForStmt(self, stmt: ForStmt):
        self.current_depth += 1
        self.var2depth[stmt.loop_var] = self.current_depth
        super().visit_ForStmt(stmt)
        self.current_depth -= 1

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.current_depth += 1
        super().visit_WhileStmt(stmt)
        self.current_depth -= 1

    def visit_LetStmt(self, stmt: LetStmt):
        for var in stmt.bind_vars:
            self.var2depth[var] = self.current_depth
        super().visit_LetStmt(stmt)

    def visit_Var(self, e: Var):
        if isinstance(e, SymbolVar):
            self.var2depth[e] = 1
        super().visit_Var(e)

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.var2depth[stmt.var] = max(self.var2depth.get(stmt.var, 0), self.current_depth)
        super().visit_AssignStmt(stmt)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.var2depth[stmt.var] = self.current_depth
        super().visit_DeclareStmt(stmt)

    def visit_Let(self, e: Let):
        raise ValueError('Please first lower the Let expression.')

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise ValueError('Please first lower the ForMappingStmt statement.')


class AdditionChainTransform:
    def __call__(self, chain: List[Expr]) -> List[Expr]:
        raise NotImplementedError()


class ReorderChain(AdditionChainTransform):
    """
    Reorder the items of the addition chain according to key(expr): (max-depth(all vars in expr), str(expr))

    The max-depth is the maximum loop depth that modified/defined the variable.

    I did some small experiments by checking which order is good for nvcc to help us generate good binary.
    Short answer is:
        we should put the loop-invariant expressions together to allow the nvcc compute and reuse it outside the loop.

    This leads to the current ordering method that put the terms with small depth (closer to outer context) at the
    beginning and put large depth to the end. Constant (with depth == 0) are always put at the end.

    For example:

    ```
    a: int = ...
    b: int = ...
    for i in range(...):
        for j in range(...):
            expr = a * i + a * b + 5 + j + i
    ```
    will be reordered to
    ```
            expr = a * b + a * i + i +  j + 5
    ```


    """

    def __init__(self, printer: IRPrinter, depth_analyzer: DepthAnalyzer):
        self.analyzer: DepthAnalyzer = depth_analyzer
        self.printer: IRPrinter = printer

    def __call__(self, chain: List[Expr]) -> List[Expr]:
        def key_func(e):
            depth = self.analyzer.get_depth(e)
            text = str(self.printer(e))
            # we order the items like
            # items with depth = 1, items with depth = 0, items with depths >= 2
            if depth == 0:
                depth = 1000
            return depth, text

        return list(sorted(chain, key=key_func))


class DivModCancellation(AdditionChainTransform):
    """
    e / c * c + e % c => e where e / c * c and e % c are two expressions in the addition chain

    expr1: e / c * c
    expr2: e % c
    """

    def __init__(self, printer):
        self.printer: IRPrinter = printer

    def __call__(self, chain: List[Expr]) -> List[Expr]:
        while True:
            replaced = self.search_and_replace(chain)
            if not replaced:
                break
        return chain

    def search_and_replace(self, chain: List[Expr]) -> bool:
        """Search for two expressions in the addition chain, and replace them with a simplified expression."""
        for e1 in chain:
            for e2 in chain:
                if e1 is e2:
                    continue
                ec = self.check(e1, e2)
                if ec is None:
                    continue
                e, c = ec
                _ = c  # unused
                chain.remove(e1)
                chain.remove(e2)
                chain.append(e)
                return True
        return False

    def check(self, e1: Expr, e2: Expr) -> Optional[Tuple[Expr, Constant]]:
        """Check if e1 and e2 are in the form of e / c * c and e % c, respectively."""
        if isinstance(e2, Mod) and isinstance(e2.b, Constant):
            e, c = e2.a, e2.b
            if (  # pylint: disable=too-many-boolean-expressions
                isinstance(e1, Multiply)
                and isinstance(e1.a, Div)
                and isinstance(e1.b, Constant)
                and isinstance(e1.a.b, Constant)
                and e1.b == e1.a.b == c
                and str(self.printer(e1.a.a)) == str(self.printer(e))
            ):
                return e, c
        return None


class AdditionChainRewriter(IRRewriter):
    def __init__(self, transform: Optional[Callable[[List[Expr]], List[Expr]]] = None):
        super().__init__()
        self.transform: Callable[[List[Expr]], List[Expr]] = transform if transform is not None else lambda x: x
        self.add2chain: Dict[Add, List[Expr]] = {}

    def visit_Var(self, e: Var):
        # the type of the variable may contain the addition expressions, but we do not modify those since
        # it will modify the variable object, our current implementation does not support alter the variable.
        return e

    def visit_Add(self, e: Add):
        # chain decomposition
        chain: List[Expr] = []
        for operand in [e.a, e.b]:
            self.visit(operand)
            if isinstance(operand, Add):
                chain.extend(self.add2chain[operand])
            else:
                chain.append(self.visit(operand))

        # chain transformation
        chain = self.transform(chain)

        # add expression reconstruction
        ret = chain[0]
        for i in range(1, len(chain)):
            ret = Add(ret, chain[i])

        self.add2chain[e] = chain
        return ret


class SimplifyAdditionChainPass(FunctionPass):
    def __init__(self):
        super().__init__()
        self.global_vars: list[Var] = []

    def process_module(self, ir_module: IRModule) -> IRModule:
        for var in ir_module.global_vars.values():
            if var.type.is_func_type():
                continue
            self.global_vars.append(var)
        return super().process_module(ir_module)

    def process_func(self, func: Function) -> Function:
        printer = IRPrinter()
        printer(func)

        # decompose expressions like (a + b)*c => a*c + b*c
        rewriter = DecomposeRewriter()
        func = rewriter(func)

        depth_analyzer = DepthAnalyzer(self.global_vars, func, printer)
        depth_analyzer.visit(func)

        # reorder order of addition
        rewriter = AdditionChainRewriter(transform=ReorderChain(printer=printer, depth_analyzer=depth_analyzer))
        func = rewriter(func)

        # cancel div-mod pair: e / c * c + e % c => e
        rewriter = AdditionChainRewriter(transform=DivModCancellation(printer=printer))
        func = rewriter(func)

        return func


def simplify_addition_chain_pass() -> FunctionPass:
    return SimplifyAdditionChainPass()
