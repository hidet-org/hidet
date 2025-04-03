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
from typing import List, Dict, Optional, Set

from hidet.ir.expr import Var, SymbolVar, Call, cast
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.dtypes import int32
from hidet.ir.functors import IRRewriter
from hidet.ir.primitives import is_primitive_function
from hidet.ir.primitives.runtime import get_symbol_value, get_ptr_symbol_value
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.tools import collect
from hidet.ir.utils.call_graph import CallGraph
from hidet.transforms import Pass


class FuncSymbols:
    def __init__(self, symbols: List[SymbolVar], symbol2param: Dict[SymbolVar, Var]):
        self.symbols: List[SymbolVar] = symbols
        self.symbol2param: Dict[SymbolVar, Var] = symbol2param


class InstantiateSymbolsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.func_symbols: Dict[str, FuncSymbols] = {}
        self.current_func: Optional[str] = None
        self.ir_module: Optional[IRModule] = None

    def visit_IRModule(self, module: IRModule):
        updated_module = module.copy().reset_funcs()
        # add the global variables (that are not the function variable) to the updated module
        updated_module.global_vars.update(
            {name: global_var for name, global_var in module.global_vars.items() if name not in module.functions}
        )
        call_graph = CallGraph(module, allow_missing=True)
        self.ir_module = updated_module

        # update the function in the reversed order of the call graph, from callee to caller
        for node in call_graph.reversed_order:
            updated_module.functions[node.func.name] = self.visit(node.func)
            # use a new memo for each function, in case there are some expressions are used in multiple functions
            self.memo.clear()

        return updated_module

    def visit_Function(self, func: Function):
        symbols: Set[SymbolVar] = set()

        if is_primitive_function(func.name):
            return func

        for node in collect(func, (SymbolVar, Call, LaunchKernelStmt)):
            if isinstance(node, SymbolVar):
                symbols.add(node)
            elif isinstance(node, (LaunchKernelStmt, Call)):
                if node.func_var.name not in self.func_symbols:
                    # calling primitive function, and all primitive functions do not use symbol vars
                    continue
                func_symbols: FuncSymbols = self.func_symbols[node.func_var.name]
                symbols.update(func_symbols.symbols)
            else:
                assert False

        ordered_symbols: List[SymbolVar] = list(symbols)
        symbol_params: List[Var] = [Var(symbol.name, symbol.type) for symbol in ordered_symbols]
        self.func_symbols[func.name] = FuncSymbols(
            symbols=ordered_symbols,
            symbol2param={symbol: param for symbol, param in zip(ordered_symbols, symbol_params)},
        )

        if len(ordered_symbols) == 0:
            # no symbol vars, no need to instantiate
            return func
        else:
            for symbol, symbol_param in zip(ordered_symbols, symbol_params):
                self.memo[symbol] = symbol_param
            self.current_func = func.name

            params = self.visit(func.params)
            body = self.visit(func.body)
            ret_type = self.visit(func.ret_type)
            attrs = self.visit(func.attrs)

            if func.kind == 'public':
                # for public function, we call the runtime primitive functions to get the symbol values
                symbol_values = []
                for symbol in ordered_symbols:
                    if symbol.type.is_data_type() and symbol.type == int32:
                        symbol_values.append(get_symbol_value(symbol.name))
                    elif symbol.type.is_pointer():
                        symbol_values.append(cast(get_ptr_symbol_value(symbol.name), dtype=symbol.type))
                    else:
                        raise NotImplementedError(symbol.type)

                if len(symbol_params) > 0:
                    body = LetStmt(bind_vars=symbol_params, bind_values=symbol_values, body=body)
            elif func.kind in [
                'cuda_kernel',
                'cuda_internal',
                'cpu_kernel',
                'cpu_internal',
                'hip_kernel',
                'hip_internal',
            ]:
                # for kernel functions, we just pass via the parameters
                params = params + symbol_params
            else:
                raise NotImplementedError()

            for symbol, symbol_param in zip(ordered_symbols, symbol_params):
                del self.memo[symbol]
            self.current_func = None

            return Function(func.name, params, body, ret_type, func.kind, attrs=attrs)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        stmt = super().visit_LaunchKernelStmt(stmt)
        if stmt.func_var.name not in self.func_symbols:
            return stmt
        if (
            stmt.func_var.name in self.ir_module.functions
            and self.ir_module.functions[stmt.func_var.name].kind == 'public'
        ):
            return stmt

        callee_func_symbols: FuncSymbols = self.func_symbols[stmt.func_var.name]
        caller_func_symbols: FuncSymbols = self.func_symbols[self.current_func]
        for callee_used_symbol in callee_func_symbols.symbols:
            assert callee_used_symbol in caller_func_symbols.symbol2param
            stmt.args.append(caller_func_symbols.symbol2param[callee_used_symbol])
        # update the function variable since the function type has changed.
        stmt.func_var = self.ir_module.lookup_var(stmt.func_var.name)
        return stmt

    def visit_Call(self, call: Call):
        call = super().visit_Call(call)
        if call.func_var.name not in self.func_symbols:
            return call
        if (
            call.func_var.name in self.ir_module.functions
            and self.ir_module.functions[call.func_var.name].kind == 'public'
        ):
            return call

        callee_func_symbols: FuncSymbols = self.func_symbols[call.func_var.name]
        caller_func_symbols: FuncSymbols = self.func_symbols[self.current_func]
        args = list(call.args)
        for callee_used_symbol in callee_func_symbols.symbols:
            assert callee_used_symbol in caller_func_symbols.symbol2param
            args.append(caller_func_symbols.symbol2param[callee_used_symbol])
        call.func_var = self.ir_module.lookup_var(call.func_var.name)
        call.args = tuple(args)
        return call


class InstantiateSymbolsPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        rewriter = InstantiateSymbolsRewriter()
        return rewriter.rewrite(ir_module)


def instantiate_symbols_pass() -> Pass:
    return InstantiateSymbolsPass()
