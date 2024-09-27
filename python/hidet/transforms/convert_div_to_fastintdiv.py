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
from typing import List, Optional, Dict
from hidet.ir.expr import Var, Div, Mod
from hidet.ir.func import Function
from hidet.ir.stmt import DeclareStmt, LetStmt
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.type import FuncType
from hidet.ir.dtypes.integer import i32
from hidet.transforms import Pass
from hidet.ir.builders import StmtBuilder
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.expr import Call, constant
from hidet.ir.primitives.runtime import calculate_magic_numbers
from hidet.logging import logger


def is_launch_function(func: Function):
    return func.kind == 'public' and 'launch' in func.name


def is_kernel_function(func: Function, func_name: str):
    return func.kind == "cuda_kernel" and func.name == func_name


def is_required_letstmt(stmt: LetStmt):
    return (
        len(stmt.bind_values) != 0
        and all(hasattr(bv, 'func_var') for bv in stmt.bind_values)
        and all(bv.func_var.name == 'get_symbol_value' for bv in stmt.bind_values)
    )


class CollectSymVarsAndFuncNames(IRVisitor):
    def __init__(self):
        super().__init__()
        self.sym_var_names: Optional[List[str]] = None
        self.kernel_function_name: Optional[str] = None

    def visit_Function(self, func: Function):
        if is_launch_function(func):
            super().visit_Function(func)
        return func

    def visit_LetStmt(self, stmt: LetStmt):
        if not is_required_letstmt(stmt):
            logger.warning(
                f'public launch function contains LetStmt {stmt} \
                         that may be optimized with fast int div'
            )
            return stmt
        super().visit_LetStmt(stmt)
        self.sym_var_names = [bv.hint for bv in stmt.bind_vars]
        return stmt

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.kernel_function_name = stmt.func_var.name
        return stmt


# This visitor creates a filtered symbol var pool that contains only
# symbol vars that are used in the kernel function as a divisor (/ symbolvar)
# or modulus (% symbolvar)
class FilterSymbolVar(IRVisitor):
    def __init__(self, sym_var_names: List[str], kernel_function_name: str):
        super().__init__()
        self.sym_var_names = sym_var_names
        self.kernel_function_name = kernel_function_name
        self.filtered_sym_var_names = set()

    def visit_Function(self, func: Function):
        if is_kernel_function(func, self.kernel_function_name):
            super().visit_Function(func)
        return func

    def visit_Div(self, e: Div):
        if isinstance(e.b, Var) and e.b.hint in self.sym_var_names:
            self.filtered_sym_var_names.add(e.b.hint)
        return super().visit_Div(e)

    def visit_Mod(self, e: Mod):
        if isinstance(e.b, Var) and e.b.hint in self.sym_var_names:
            self.filtered_sym_var_names.add(e.b.hint)
        return super().visit_Mod(e)


class GenerateMagicVarsRewriter(IRRewriter):
    def __init__(self, filtered_sym_var_names: List[str]):
        super().__init__()
        self.magic_vars: Dict[str, List[Var]] = {}
        self.filtered_sym_var_names = filtered_sym_var_names

    def visit_Function(self, func: Function):
        if is_launch_function(func):
            new_func = super().visit_Function(func)
            return new_func
        return func

    def visit_LetStmt(self, stmt: LetStmt):
        if not is_required_letstmt(stmt):
            return stmt
        sb = StmtBuilder()
        for bind_var in stmt.bind_vars:
            if bind_var.hint not in self.filtered_sym_var_names:
                continue
            magic_m = Var(f'magic_m_{bind_var.hint}', bind_var.type, f'magic_m_{bind_var.hint}')
            magic_s = Var(f'magic_s_{bind_var.hint}', bind_var.type, f'magic_s_{bind_var.hint}')
            magic_as = Var(f'magic_as_{bind_var.hint}', bind_var.type, f'magic_as_{bind_var.hint}')
            sb += DeclareStmt(magic_m, constant(0, i32))
            sb += DeclareStmt(magic_s, constant(0, i32))
            sb += DeclareStmt(magic_as, constant(0, i32))
            sb += calculate_magic_numbers(bind_var, magic_m, magic_s, magic_as)
            self.magic_vars[bind_var.hint] = [magic_m, magic_s, magic_as]
        super().visit_LetStmt(stmt)
        sb += stmt.body
        stmt.body = sb.finish()
        return LetStmt(stmt.bind_vars, stmt.bind_values, stmt.body)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        if self.magic_vars:
            stmt.args = stmt.args + [item for sublist in list(self.magic_vars.values()) for item in sublist]
            return LaunchKernelStmt(
                stmt.func_var, stmt.args, stmt.grid_dim, stmt.cluster_dim, stmt.block_dim, stmt.shared_mem_bytes
            )
        else:
            return stmt


class ExpandFunctionParamRewriter(IRRewriter):
    def __init__(self, magic_vars: Dict[str, List[Var]], kernel_function_name: str):
        super().__init__()
        self.magic_vars = magic_vars
        self.kernel_function_name = kernel_function_name

    def visit_Function(self, func: Function):
        if is_kernel_function(func, self.kernel_function_name):
            func = super().visit_Function(func)
            func.params = func.params + [item for sublist in list(self.magic_vars.values()) for item in sublist]
            return Function(func.name, func.params, func.body, func.ret_type, func.kind, func.attrs)
        return func

    def visit_Div(self, e: Div):
        if isinstance(e.b, Var) and e.b.hint in self.magic_vars.keys():
            fastdiv_prim = Var('fastintdiv', FuncType([i32, i32, i32, i32, i32], i32), 'fastintdiv')
            return Call(
                fastdiv_prim,
                (e.a, e.b, self.magic_vars[e.b.hint][0], self.magic_vars[e.b.hint][1], self.magic_vars[e.b.hint][2]),
            )
        return super().visit_Div(e)

    def visit_Mod(self, e: Mod):
        if isinstance(e.b, Var) and e.b.hint in self.magic_vars.keys():
            fastmod_prim = Var('fastintmod', FuncType([i32, i32, i32, i32, i32], i32), 'fastintmod')
            return Call(
                fastmod_prim,
                (e.a, e.b, self.magic_vars[e.b.hint][0], self.magic_vars[e.b.hint][1], self.magic_vars[e.b.hint][2]),
            )
        return super().visit_Mod(e)


class ConvertDivToFastIntDivPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        collector = CollectSymVarsAndFuncNames()
        collector.visit(ir_module)
        if collector.sym_var_names is None or collector.kernel_function_name is None:
            return ir_module
        filter = FilterSymbolVar(collector.sym_var_names, collector.kernel_function_name)
        filter.visit(ir_module)
        if not filter.filtered_sym_var_names:
            return ir_module
        generate_rewriter = GenerateMagicVarsRewriter(list(filter.filtered_sym_var_names))
        ir_module = generate_rewriter.visit(ir_module)
        expand_rewriter = ExpandFunctionParamRewriter(generate_rewriter.magic_vars, collector.kernel_function_name)
        ir_module = expand_rewriter.visit(ir_module)
        return ir_module


def convert_div_to_fastintdiv_pass() -> Pass:
    return ConvertDivToFastIntDivPass()
