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
from typing import List, Dict, Set

from hidet.ir.expr import Call, Var, Expr
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import Stmt, SeqStmt, DeclareStmt, EvaluateStmt, ReturnStmt
from hidet.ir.tools import collect, rewrite
from hidet.ir.type import ReferenceType, TensorType
from hidet.ir.utils.call_graph import CallGraph, CallGraphNode
from hidet.transforms import Pass


class InlineFunctionRewriter(IRRewriter):
    def __init__(self, updated_ir_module: IRModule):
        super().__init__(use_memo=False)
        self.ir_module = updated_ir_module
        self.stmts: List[Stmt] = []
        self.should_inline_cache: Dict[Function, bool] = {}

    def should_inline(self, callee: Function):
        """
        Check if a function should be inlined.

        Currently, we only inline functions that
        1. have no return value
        2. have no return statement
        3. have no reference type and tensor type arguments

        Parameters
        ----------
        callee: Function
            The function to be checked

        Returns
        -------
        ret: bool
            True if the function should be inlined
        """
        if callee in self.should_inline_cache:
            return self.should_inline_cache[callee]

        if not callee.ret_type.is_void():
            ret = False
        elif callee.kind in ['packed_func', 'host_kernel', 'cuda_kernel']:
            ret = False
        elif any(isinstance(arg.type, (ReferenceType, TensorType)) for arg in callee.params):
            ret = False
        elif len(collect(callee.body, ReturnStmt)) > 0:
            ret = False
        else:
            ret = True
        self.should_inline_cache[callee] = ret
        return ret

    def inline(self, caller: Function) -> Function:
        return self.visit(caller)

    def visit(self, node):
        if isinstance(node, Stmt):
            ret = super().visit(node)
            if len(self.stmts) > 0:
                # the inlined statements should be inserted before the current statement
                ret = SeqStmt(self.stmts + [ret])
                self.stmts.clear()
            return ret
        else:
            return super().visit(node)

    def visit_Call(self, e: Call):
        if e.func_var.hint not in self.ir_module.functions:
            # primitive function that has not been imported yet
            return super().visit_Call(e)

        callee: Function = self.ir_module.functions[e.func_var.hint]
        if self.should_inline(callee):
            assert len(e.args) == len(callee.params)
            args: List[Expr] = [self.visit(arg) for arg in e.args]
            param_vars: List[Var] = []
            remap: Dict[Var, Expr] = {}
            for arg, param in zip(args, callee.params):
                if isinstance(arg, Var) and arg.type.is_tensor():
                    remap[param] = arg
                else:
                    param_var = Var(param.hint, rewrite(param.type, remap, clone_internal_var=True))
                    param_vars.append(param_var)
                    self.stmts.append(DeclareStmt(param_var, init=arg))
                    remap[param] = param_var
            callee_body = rewrite(callee.body, remap, clone_internal_var=True)
            self.stmts.append(callee_body)
            return None
        else:
            return super().visit_Call(e)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        expr = self.visit(stmt.expr)
        if expr is None:
            return SeqStmt([])
        else:
            if expr is stmt.expr:
                return stmt
            else:
                return EvaluateStmt(expr)


def inline_callees(caller: Function, updated_ir_module: IRModule) -> Function:
    rewriter = InlineFunctionRewriter(updated_ir_module)
    return rewriter.inline(caller)


class PruneUnusedFunctionRewriter(IRRewriter):
    def visit_IRModule(self, module: IRModule):
        call_graph = CallGraph(module, allow_missing=True)
        unused_func_names: Set[str] = set()
        for node in call_graph.nodes:
            func: Function = node.func
            if func.kind in ['packed_func', 'host_kernel', 'cuda_kernel']:
                continue
            if len(node.callers) == 0:
                unused_func_names.add(func.name)
        for func_name in unused_func_names:
            del module.functions[func_name]
            if func_name in module.global_vars:
                del module.global_vars[func_name]

        return module


def prune_unused_functions(ir_module: IRModule):
    rewriter = PruneUnusedFunctionRewriter()
    return rewriter.visit(ir_module)


class InlineFunctionPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        call_graph = CallGraph(ir_module, allow_missing=True)
        updated_ir_module = IRModule(task=ir_module.task)
        for node in call_graph.reversed_order:
            assert isinstance(node, CallGraphNode)
            func = inline_callees(node.func, updated_ir_module)
            updated_ir_module.functions[func.name] = func

        updated_ir_module = prune_unused_functions(updated_ir_module)

        return updated_ir_module


def inline_function_pass():
    return InlineFunctionPass()
