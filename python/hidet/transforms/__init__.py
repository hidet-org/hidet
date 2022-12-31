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
from hidet.ir.func import IRModule

from .base import Pass, FunctionPass, FunctionBodyPass, SequencePass, RepeatFunctionPass, PassContext
from .instruments import PassInstrument, SaveIRInstrument, ProfileInstrument

from .flatten_tensor_slice import flatten_tensor_slice_pass
from .flatten_tensor_index import flatten_tensor_index_pass
from .generate_packed_func import generate_packed_func_pass
from .import_primitive_functions import import_primitive_functions_pass
from .simplify_stmt import simplify_stmt_pass
from .expand_let_expr import expand_let_expr_pass
from .resolve_generic_primitive_function import resolve_primitive_func_pass
from .add_explicit_cast import add_explicit_cast_pass
from .inline_let_stmt import inline_let_stmt_pass
from .rule_based_simplifier import rule_based_simplify_pass
from .normalize_const_tensor import normalize_const_tensor_pass
from .lower_task_mapping import lower_task_mapping_pass
from .lower_protect_access import lower_protect_access_pass
from .declare_to_let import declare_to_let_pass
from .propogate_launch_bound import propagate_launch_bound_pass
from .lower_special_cast import lower_special_cast_pass


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        # necessary passes
        flatten_tensor_slice_pass(),
        lower_protect_access_pass(),
        lower_task_mapping_pass(),
        normalize_const_tensor_pass(),
        rule_based_simplify_pass(),  # make ir more readable
        flatten_tensor_index_pass(),
        lower_special_cast_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        propagate_launch_bound_pass(),
        add_explicit_cast_pass(),
        declare_to_let_pass(),
        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(inline_all=False),
        rule_based_simplify_pass(),
        simplify_stmt_pass(),
    ]

    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)

    return ir_module
